"""
coastal_validation_executor.py — Golden File Validation against TerraME CSVs
=============================================================================

Validation executor for the BR-MANGUE coastal dynamics model.
Mirrors the pattern of LuccBenchmarkExecutor, adapted for:
  - 30 per-step golden CSV files (instead of a single DBF)
  - 3 bands: uso (int), solo (int), alt (float)
  - Two comparison axes: Vector vs TerraME, Raster vs TerraME
  - One internal axis:   Vector vs Raster

Input contract
--------------
  record.source.uri          → path/URI to the input shapefile (elevacao_pol.shp)
  record.parameters["golden_dir"] → path to directory containing step_01.csv … step_NN.csv
  record.parameters["taxa_elevacao"] → float, default 0.05
  record.parameters["altura_mare"]   → float, default 6.0
  record.parameters["end_time"]      → int,   default 30
  record.parameters["checkpoints"]   → list[int], steps to validate; default [1,5,10,15,20,25,30]
  record.parameters["alt_atol"]      → float, absolute tolerance for alt; default 1e-3

Output artifacts
----------------
  scatter.png  — 3×N_CHECKPOINTS grid: Vector/TerraME, Raster/TerraME, Vector/Raster
  report.md    — per-step accuracy table + runtime summary

Usage (CLI)
-----------
    python coastal_validation_executor.py \\
        --input_dataset data/elevacao_pol.shp \\
        --parameters '{"golden_dir":"output/golden","end_time":30}'
"""
from __future__ import annotations

import io
import pathlib
import time

import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from dissmodel.core               import Environment
from dissmodel.executor           import ExperimentRecord, ModelExecutor
from dissmodel.executor.cli       import run_cli
from dissmodel.geo.raster.backend import RasterBackend
from dissmodel.io                 import load_dataset
from dissmodel.io._utils          import write_bytes, write_text
from dissmodel.executor.config    import settings

from coastal_dynamics.raster.flood_model    import FloodModel    as RasterFlood
from coastal_dynamics.raster.mangrove_model import MangroveModel as RasterMangue
from coastal_dynamics.vector.flood_model    import FloodModel    as VectorFlood
from coastal_dynamics.vector.mangrove_model import MangroveModel as VectorMangue
from coastal_dynamics.common.constants      import CRS, CELL_SIZE

# Bands and their comparison strategy
BANDS: dict[str, str] = {
    "uso":  "exact",    # integer — must match perfectly
    "solo": "exact",    # integer — must match perfectly
    "alt":  "approx",   # float   — tolerance-based
}

DEFAULT_CHECKPOINTS = [1, 5, 10, 15, 20, 25, 30]


class CoastalValidationExecutor(ModelExecutor):
    """
    Validation executor for BR-MANGUE coastal dynamics.

    Runs Vector and Raster models against the same input shapefile,
    then compares both against TerraME golden CSV files step by step.

    Three comparison axes:
        Vector vs TerraME — fidelity of vector substrate to original Lua
        Raster vs TerraME — fidelity of raster substrate to original Lua
        Vector vs Raster  — substrate equivalence (independent of TerraME)
    """

    name = "coastal_validation_terrame"

    # ── public contract ───────────────────────────────────────────────────────

    def load(
        self, record: ExperimentRecord
    ) -> tuple[gpd.GeoDataFrame, dict[int, pd.DataFrame]]:
        """
        Load input shapefile and all golden CSV files.

        Returns (gdf, golden_map) where golden_map maps step → DataFrame
        sorted by (row, col) with canonical lowercase column names.
        """
        # ── shapefile ─────────────────────────────────────────────────────────
        gdf, checksum = load_dataset(record.source.uri, fmt="vector")
        record.source.checksum = checksum

        if record.column_map:
            gdf = gdf.rename(columns={v: k for k, v in record.column_map.items()})

        gdf.columns = [c.lower() for c in gdf.columns]
        gdf = gdf.sort_values(["row", "col"]).reset_index(drop=True)

        record.add_log(
            f"Loaded shapefile: {len(gdf):,} cells  crs={gdf.crs}"
        )

        # ── golden CSVs ───────────────────────────────────────────────────────
        golden_dir  = pathlib.Path(record.parameters["golden_dir"])
        checkpoints = record.parameters.get("checkpoints", DEFAULT_CHECKPOINTS)

        golden_map: dict[int, pd.DataFrame] = {}
        for step in checkpoints:
            path = golden_dir / f"step_{step:02d}.csv"
            if not path.exists():
                raise FileNotFoundError(
                    f"Golden file not found: {path}\n"
                    f"Run the TerraME model first and place the CSVs in {golden_dir}"
                )
            df = pd.read_csv(path)
            df.columns = [c.lower() for c in df.columns]
            df = df.sort_values(["row", "col"]).reset_index(drop=True)
            golden_map[step] = df
            record.add_log(f"  Loaded golden step {step:02d}: {len(df):,} rows")

        return gdf, golden_map

    def validate(self, record: ExperimentRecord) -> None:
        """Stateless pre-flight checks — no data loading."""
        _normalize_params(record.parameters)

        if not record.source.uri:
            raise ValueError(
                "source.uri is empty — pass the input shapefile path."
            )
        if "golden_dir" not in record.parameters:
            raise ValueError(
                "Missing required parameter 'golden_dir' "
                "(path to directory with step_NN.csv files)."
            )
        checkpoints = record.parameters.get("checkpoints", DEFAULT_CHECKPOINTS)
        end_time    = record.parameters.get("end_time", 30)
        invalid     = [s for s in checkpoints if s < 1 or s > end_time]
        if invalid:
            raise ValueError(
                f"checkpoints {invalid} are outside [1, {end_time}]. "
                f"Adjust 'checkpoints' or 'end_time'."
            )

    def run(
        self,
        data: tuple[gpd.GeoDataFrame, dict[int, pd.DataFrame]],
        record: ExperimentRecord,
    ) -> dict:
        """
        Run both substrates and compare against TerraME golden files.

        `data` is the (gdf, golden_map) tuple returned by load(),
        injected by the platform. No I/O happens here.
        """
        _normalize_params(record.parameters)
        params        = record.parameters
        end_time      = params.get("end_time",      30)
        taxa_elevacao = params.get("taxa_elevacao",  0.05)
        altura_mare   = params.get("altura_mare",    6.0)
        checkpoints   = params.get("checkpoints",    DEFAULT_CHECKPOINTS)
        alt_atol      = params.get("alt_atol",       1e-3)

        gdf_orig, golden_map = data

        # ── vector run ────────────────────────────────────────────────────────
        record.add_log(f"Running Vector Model (1 → {end_time})...")
        gdf_vec = gdf_orig.copy()
        env_vec = Environment(start_time=1, end_time=end_time)
        VectorFlood(
            gdf           = gdf_vec,
            taxa_elevacao = taxa_elevacao,
        )
        VectorMangue(
            gdf           = gdf_vec,
            taxa_elevacao = taxa_elevacao,
            altura_mare   = altura_mare,
        )
        t0     = time.perf_counter()
        env_vec.run()
        vec_ms = (time.perf_counter() - t0) * 1000 / end_time
        record.add_log(f"Vector done: {vec_ms:.1f} ms/step")

        # ── raster run ────────────────────────────────────────────────────────
        record.add_log(f"Running Raster Model (1 → {end_time})...")
        backend, rows_idx, cols_idx = _build_raster(gdf_orig)
        env_ras = Environment(start_time=1, end_time=end_time)
        RasterFlood(
            backend       = backend,
            taxa_elevacao = taxa_elevacao,
        )
        RasterMangue(
            backend       = backend,
            taxa_elevacao = taxa_elevacao,
            altura_mare   = altura_mare,
        )
        t0     = time.perf_counter()
        env_ras.run()
        ras_ms = (time.perf_counter() - t0) * 1000 / end_time
        record.add_log(f"Raster done: {ras_ms:.1f} ms/step")

        # ── metrics per checkpoint ────────────────────────────────────────────
        record.add_log("Calculating metrics...")
        metrics: dict[int, dict] = {}

        for step in checkpoints:
            golden    = golden_map[step]
            step_data = {}

            for band, strategy in BANDS.items():
                if band not in gdf_vec.columns or band not in golden.columns:
                    continue

                vec_vals  = gdf_vec[band].values.astype(float)
                ras_vals  = backend.get(band)[rows_idx, cols_idx].astype(float)
                gold_vals = golden[band].values.astype(float)
                tol       = alt_atol if strategy == "approx" else 0.0

                step_data[band] = {
                    "Vector_vs_TerraME": _metrics(vec_vals,  gold_vals, tol),
                    "Raster_vs_TerraME": _metrics(ras_vals,  gold_vals, tol),
                    "Vector_vs_Raster":  _metrics(vec_vals,  ras_vals,  tol),
                    # keep raw arrays for scatter plots
                    "_vec":  vec_vals,
                    "_ras":  ras_vals,
                    "_gold": gold_vals,
                }

            metrics[step] = step_data
            for band, bdata in step_data.items():
                for label, m in bdata.items():
                    if label.startswith("_"):
                        continue
                    record.add_log(
                        f"  step={step:02d}  {band}  {label}: "
                        f"match={m['match_pct']:.1f}%  MAE={m['mae']:.5f}"
                    )

        # ── scatter plots ─────────────────────────────────────────────────────
        record.add_log("Generating scatter plots...")
        buf = _make_scatter(metrics, checkpoints, end_time)

        return {
            "plot_buf":    buf,
            "report_str":  _build_markdown(
                end_time, checkpoints, alt_atol, vec_ms, ras_ms, metrics
            ),
            "metrics":     {
                str(step): {
                    band: {k: v for k, v in bdata.items() if not k.startswith("_")}
                    for band, bdata in step_data.items()
                }
                for step, step_data in metrics.items()
            },
        }

    def save(self, result: dict, record: ExperimentRecord) -> ExperimentRecord:
        """Write scatter.png and report.md to output_path."""
        base_uri = (
            record.output_path
            or f"{settings.default_output_base}/experiments/"
               f"{record.experiment_id}/coastal_validation"
        )

        record.add_artifact(
            "plot",
            write_bytes(
                result["plot_buf"],
                f"{base_uri}/scatter.png",
                content_type="image/png",
            ),
        )
        record.add_artifact(
            "report",
            write_text(
                result["report_str"],
                f"{base_uri}/report.md",
                content_type="text/markdown",
            ),
        )

        record.output_path = base_uri
        record.metrics     = result["metrics"]
        record.status      = "completed"
        record.add_log(f"Saved scatter.png → {base_uri}/scatter.png")
        record.add_log(f"Saved report.md   → {base_uri}/report.md")
        return record


# ── helpers ───────────────────────────────────────────────────────────────────
def _normalize_params(params: dict) -> None:
    """
    Coerce CLI string values to their expected Python types.
    The run_cli parser delivers all --param values as strings;
    this function converts them in-place before any model logic runs.
    """
    import ast
    for key in ("end_time",):
        if key in params and isinstance(params[key], str):
            params[key] = int(params[key])
    for key in ("taxa_elevacao", "altura_mare", "alt_atol"):
        if key in params and isinstance(params[key], str):
            params[key] = float(params[key])
    if "checkpoints" in params and isinstance(params["checkpoints"], str):
        params["checkpoints"] = [int(x) for x in ast.literal_eval(params["checkpoints"])]



def _metrics(a: np.ndarray, b: np.ndarray, tol: float) -> dict:
    diff = np.abs(a - b)
    return {
        "match_pct": float((diff <= tol).mean() * 100),
        "mae":       float(diff.mean()),
        "rmse":      float(np.sqrt((diff ** 2).mean())),
        "max_err":   float(diff.max()),
        "n_cells":   len(a),
    }


def _build_raster(
    gdf: gpd.GeoDataFrame,
) -> tuple[RasterBackend, np.ndarray, np.ndarray]:
    from dissmodel.io.convert import vector_to_raster_backend

    backend = vector_to_raster_backend(
        source      = gdf,
        resolution  = CELL_SIZE,
        attrs       = {"uso": 0, "alt": 0.0, "solo": 0},
        crs         = CRS,
        all_touched = False,
        nodata      = 0,
    )
    rows = gdf["row"].astype(int).values
    cols = gdf["col"].astype(int).values

    # normalise to 0-based — shapefile indices may start at 1
    rows = rows - rows.min()
    cols = cols - cols.min()

    return backend, rows, cols


def _make_scatter(
    metrics:     dict[int, dict],
    checkpoints: list[int],
    end_time:    int,
) -> io.BytesIO:
    """
    Build a scatter grid: rows = bands, cols = comparison axes.
    Each cell shows the final checkpoint only — cleaner than one plot per step.
    Uses the last checkpoint as the representative state.
    """
    last_step = checkpoints[-1]
    step_data = metrics[last_step]
    bands     = [b for b in BANDS if b in step_data]
    axes_labels = ["Vector_vs_TerraME", "Raster_vs_TerraME", "Vector_vs_Raster"]
    axis_titles = {
        "Vector_vs_TerraME": "Vector vs TerraME",
        "Raster_vs_TerraME": "Raster vs TerraME",
        "Vector_vs_Raster":  "Vector vs Raster",
    }
    x_labels = {
        "Vector_vs_TerraME": ("Vector",  "TerraME"),
        "Raster_vs_TerraME": ("Raster",  "TerraME"),
        "Vector_vs_Raster":  ("Vector",  "Raster"),
    }
    raw_map = {
        "Vector_vs_TerraME": ("_vec",  "_gold"),
        "Raster_vs_TerraME": ("_ras",  "_gold"),
        "Vector_vs_Raster":  ("_vec",  "_ras"),
    }

    n_rows = len(bands)
    n_cols = len(axes_labels)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))

    # ensure 2D indexing even with 1 band
    if n_rows == 1:
        axes = [axes]

    for r, band in enumerate(bands):
        bdata = step_data[band]
        for c, axis_key in enumerate(axes_labels):
            ax  = axes[r][c]
            m   = bdata[axis_key]
            xk, yk = raw_map[axis_key]
            x   = bdata[xk]
            y   = bdata[yk]
            xl, yl = x_labels[axis_key]

            ax.scatter(x, y, alpha=0.3, s=4, color="steelblue")
            lim = max(float(np.max(x)), float(np.max(y))) * 1.05 or 1.0
            ax.plot([0, lim], [0, lim], "r--", lw=1)
            ax.set_xlabel(f"{xl} {band}")
            ax.set_ylabel(f"{yl} {band}")
            ax.set_title(f"{band} — {axis_titles[axis_key]}")
            ax.text(
                0.05, 0.85,
                f"Match={m['match_pct']:.1f}%\n"
                f"MAE={m['mae']:.5f}\n"
                f"RMSE={m['rmse']:.5f}",
                transform = ax.transAxes,
                fontsize  = 7,
                bbox      = dict(boxstyle="round", facecolor="wheat", alpha=0.5),
            )

    plt.suptitle(
        f"BR-MANGUE Validation — step {last_step}/{end_time}",
        fontsize=12,
    )
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=150)
    plt.close()
    buf.seek(0)
    return buf


def _build_markdown(
    end_time:    int,
    checkpoints: list[int],
    alt_atol:    float,
    vec_ms:      float,
    ras_ms:      float,
    metrics:     dict[int, dict],
) -> str:
    speedup = vec_ms / ras_ms if ras_ms > 0 else float("inf")
    lines = [
        "# BR-MANGUE Validation Report\n\n",
        f"**Steps:** 1 → {end_time} | **Alt tolerance:** {alt_atol} m\n\n",
        "## Runtime\n\n",
        "| Substrate | ms/step | Speedup |\n|---|---|---|\n",
        f"| Vector | {vec_ms:.1f} | 1.0× |\n",
        f"| Raster | {ras_ms:.1f} | {speedup:.1f}× |\n\n",
        "## Accuracy per band\n\n",
    ]

    for band in BANDS:
        lines.append(f"### `{band}`\n\n")
        lines.append(
            "| Step | Comparison | Match % | MAE | RMSE | Max err |\n"
            "|---|---|---|---|---|---|\n"
        )
        for step in checkpoints:
            step_data = metrics.get(step, {})
            if band not in step_data:
                continue
            bdata = step_data[band]
            for label, m in bdata.items():
                if label.startswith("_"):
                    continue
                lines.append(
                    f"| {step:02d} | {label.replace('_', ' ')} | "
                    f"{m['match_pct']:.2f}% | {m['mae']:.6f} | "
                    f"{m['rmse']:.6f} | {m['max_err']:.6f} |\n"
                )
        lines.append("\n")

    return "".join(lines)


if __name__ == "__main__":
    run_cli(CoastalValidationExecutor)
