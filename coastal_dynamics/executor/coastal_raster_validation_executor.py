"""
coastal_raster_validation_executor.py — Raster vs TerraME Validation
=====================================================================

Simplified validation executor: runs only the Raster substrate and
compares it against TerraME golden CSV files step by step.

Input contract
--------------
  record.source.uri                  → path to input shapefile or zip
  record.parameters["golden_dir"]    → directory with step_NN.csv files
  record.parameters["end_time"]      → int,   default 30
  record.parameters["taxa_elevacao"] → float, default 0.05
  record.parameters["altura_mare"]   → float, default 6.0
  record.parameters["checkpoints"]   → list[int], default [1,5,10,15,20,25,30]
  record.parameters["alt_atol"]      → float, default 1e-3

Output artifacts
----------------
  scatter.png  — 3 scatter plots (uso, solo, alt) at the last checkpoint
  report.md    — per-step accuracy table + runtime

Usage
-----
    python coastal_dynamics/executor/coastal_raster_validation_executor.py run \\
      --input  examples/data/input/elevacao_pol.zip \\
      --output examples/data/output/validation \\
      --param  golden_dir=tests/fixtures/golden \\
      --param  end_time=30 \\
      --param  taxa_elevacao=0.05 \\
      --param  altura_mare=6.0 \\
      --param  checkpoints=[1,5,10,15,20,25,30]
"""
from __future__ import annotations

import ast
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
from dissmodel.io                 import load_dataset
from dissmodel.io._utils          import write_bytes, write_text
from dissmodel.executor.config    import settings

from coastal_dynamics.raster.flood_model    import FloodModel    as RasterFlood
from coastal_dynamics.raster.mangrove_model import MangroveModel as RasterMangue
from coastal_dynamics.common.constants      import CRS, CELL_SIZE

BANDS: dict[str, str] = {
    "uso":  "exact",
    "solo": "exact",
    "alt":  "approx",
}

DEFAULT_CHECKPOINTS = [1, 5, 10, 15, 20, 25, 30]


class CoastalRasterValidationExecutor(ModelExecutor):
    """
    Raster-only validation executor for BR-MANGUE coastal dynamics.

    Loads the input shapefile, rasterises it, runs FloodModel +
    MangroveModel, and compares the result against TerraME golden CSVs.
    """

    name = "coastal_raster_validation"

    # ── public contract ───────────────────────────────────────────────────────

    def load(
        self, record: ExperimentRecord
    ) -> tuple[gpd.GeoDataFrame, dict[int, pd.DataFrame]]:
        _normalize_params(record.parameters)
        checkpoints = record.parameters.get("checkpoints", DEFAULT_CHECKPOINTS)

        # ── shapefile ─────────────────────────────────────────────────────────
        gdf, checksum = load_dataset(record.source.uri, fmt="vector")
        record.source.checksum = checksum

        if record.column_map:
            gdf = gdf.rename(columns={v: k for k, v in record.column_map.items()})

        gdf.columns = [c.lower() for c in gdf.columns]
        gdf = gdf.sort_values(["row", "col"]).reset_index(drop=True)
        record.add_log(f"Loaded shapefile: {len(gdf):,} cells  crs={gdf.crs}")

        # ── golden CSVs ───────────────────────────────────────────────────────
        golden_dir = pathlib.Path(record.parameters["golden_dir"])
        golden_map: dict[int, pd.DataFrame] = {}

        for step in checkpoints:
            path = golden_dir / f"step_{step:02d}.csv"
            if not path.exists():
                raise FileNotFoundError(
                    f"Golden file not found: {path}\n"
                    f"Run the TerraME model first and place CSVs in {golden_dir}"
                )
            df = pd.read_csv(path)
            df.columns = [c.lower() for c in df.columns]
            df = df.sort_values(["row", "col"]).reset_index(drop=True)
            golden_map[step] = df
            record.add_log(f"  Loaded golden step {step:02d}: {len(df):,} rows")

        return gdf, golden_map

    def validate(self, record: ExperimentRecord) -> None:
        _normalize_params(record.parameters)

        if not record.source.uri:
            raise ValueError("source.uri is empty — pass the input shapefile.")

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
        _normalize_params(record.parameters)
        params        = record.parameters
        end_time      = params.get("end_time",      30)
        taxa_elevacao = params.get("taxa_elevacao",  0.05)
        altura_mare   = params.get("altura_mare",    6.0)
        checkpoints   = params.get("checkpoints",    DEFAULT_CHECKPOINTS)
        alt_atol      = params.get("alt_atol",       1e-3)

        gdf_orig, golden_map = data

        # ── raster run ────────────────────────────────────────────────────────
        record.add_log(f"Running Raster Model (1 → {end_time})...")
        backend, rows_idx, cols_idx = _build_raster(gdf_orig)

        env = Environment(start_time=1, end_time=end_time)
        RasterFlood(backend=backend, taxa_elevacao=taxa_elevacao)
        RasterMangue(backend=backend, taxa_elevacao=taxa_elevacao, altura_mare=altura_mare)

        t0     = time.perf_counter()
        env.run()
        ras_ms = (time.perf_counter() - t0) * 1000 / end_time
        record.add_log(f"Raster done: {ras_ms:.1f} ms/step")

        # ── metrics per checkpoint ────────────────────────────────────────────
        record.add_log("Calculating metrics...")
        metrics: dict[int, dict] = {}

        for step in checkpoints:
            golden    = golden_map[step]
            step_data = {}

            for band, strategy in BANDS.items():
                if band not in golden.columns:
                    continue

                ras_vals  = backend.get(band)[rows_idx, cols_idx].astype(float)
                gold_vals = golden[band].values.astype(float)
                tol       = alt_atol if strategy == "approx" else 0.0

                step_data[band] = {
                    "metrics": _metrics(ras_vals, gold_vals, tol),
                    "_ras":    ras_vals,
                    "_gold":   gold_vals,
                }

            metrics[step] = step_data
            for band, bdata in step_data.items():
                m = bdata["metrics"]
                record.add_log(
                    f"  step={step:02d}  {band}: "
                    f"match={m['match_pct']:.1f}%  "
                    f"MAE={m['mae']:.5f}  "
                    f"max_err={m['max_err']:.5f}"
                )

        # ── scatter plots ─────────────────────────────────────────────────────
        record.add_log("Generating scatter plots...")
        buf = _make_scatter(metrics, checkpoints, end_time)

        return {
            "plot_buf":   buf,
            "report_str": _build_markdown(
                end_time, checkpoints, alt_atol, ras_ms, metrics
            ),
            "metrics": {
                str(step): {
                    band: bdata["metrics"]
                    for band, bdata in step_data.items()
                }
                for step, step_data in metrics.items()
            },
        }

    def save(self, result: dict, record: ExperimentRecord) -> ExperimentRecord:
        base_uri = (
            record.output_path
            or f"{settings.default_output_base}/experiments/"
               f"{record.experiment_id}/raster_validation"
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
    """Coerce CLI string values to their expected Python types."""
    for key in ("end_time",):
        if key in params and isinstance(params[key], str):
            params[key] = int(params[key])
    for key in ("taxa_elevacao", "altura_mare", "alt_atol"):
        if key in params and isinstance(params[key], str):
            params[key] = float(params[key])
    if "checkpoints" in params and isinstance(params["checkpoints"], str):
        params["checkpoints"] = [
            int(x) for x in ast.literal_eval(params["checkpoints"])
        ]


def _metrics(a: np.ndarray, b: np.ndarray, tol: float) -> dict:
    diff = np.abs(a - b)
    return {
        "match_pct": float((diff <= tol).mean() * 100),
        "mae":       float(diff.mean()),
        "rmse":      float(np.sqrt((diff ** 2).mean())),
        "max_err":   float(diff.max()),
        "n_cells":   len(a),
    }


def _build_raster(gdf: gpd.GeoDataFrame):
    """
    Build a RasterBackend aligned 1:1 with the GDF using row/col attributes.

    Uses the shapefile's own row/col grid indices (normalised to 0-based)
    as array coordinates — bypasses geographic rasterisation and guarantees
    that backend[rows_idx, cols_idx] maps back to the correct GDF rows.
    """
    from dissmodel.geo.raster.backend import RasterBackend

    rows = gdf["row"].astype(int).values
    cols = gdf["col"].astype(int).values

    # normalise to 0-based — shapefile indices may start at 1
    rows = rows - rows.min()
    cols = cols - cols.min()

    n_rows = int(rows.max()) + 1
    n_cols = int(cols.max()) + 1

    backend = RasterBackend(shape=(n_rows, n_cols))

    # mask: only cells present in the GDF are valid
    mask = np.zeros((n_rows, n_cols), dtype=bool)
    mask[rows, cols] = True
    backend.set("mask", mask)

    for band in ("uso", "alt", "solo"):
        if band in gdf.columns:
            arr = np.zeros((n_rows, n_cols), dtype=np.float32)
            arr[rows, cols] = gdf[band].astype(float).values
            backend.set(band, arr)

    return backend, rows, cols


def _make_scatter(
    metrics:     dict[int, dict],
    checkpoints: list[int],
    end_time:    int,
) -> io.BytesIO:
    """One scatter plot per band (Raster vs TerraME) at the last checkpoint."""
    last_step = checkpoints[-1]
    step_data = metrics[last_step]
    bands     = [b for b in BANDS if b in step_data]

    fig, axes = plt.subplots(1, len(bands), figsize=(5 * len(bands), 4))
    if len(bands) == 1:
        axes = [axes]

    for ax, band in zip(axes, bands):
        bdata = step_data[band]
        m     = bdata["metrics"]
        ras   = bdata["_ras"]
        gold  = bdata["_gold"]

        ax.scatter(ras, gold, alpha=0.3, s=4, color="steelblue")
        lim = max(float(np.max(ras)), float(np.max(gold))) * 1.05 or 1.0
        ax.plot([0, lim], [0, lim], "r--", lw=1)
        ax.set_xlabel(f"Raster {band}")
        ax.set_ylabel(f"TerraME {band}")
        ax.set_title(f"{band}")
        ax.text(
            0.05, 0.80,
            f"Match={m['match_pct']:.1f}%\n"
            f"MAE={m['mae']:.5f}\n"
            f"RMSE={m['rmse']:.5f}\n"
            f"N={m['n_cells']:,}",
            transform = ax.transAxes,
            fontsize  = 7,
            bbox      = dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

    plt.suptitle(
        f"BR-MANGUE — Raster vs TerraME — step {last_step}/{end_time}",
        fontsize=11,
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
    ras_ms:      float,
    metrics:     dict[int, dict],
) -> str:
    lines = [
        "# BR-MANGUE — Raster vs TerraME Validation Report\n\n",
        f"**Steps:** 1 → {end_time} | "
        f"**Runtime:** {ras_ms:.1f} ms/step | "
        f"**Alt tolerance:** {alt_atol} m\n\n",
        "## Accuracy per band\n\n",
    ]

    for band in BANDS:
        lines.append(f"### `{band}`\n\n")
        lines.append(
            "| Step | Match % | MAE | RMSE | Max err | N cells |\n"
            "|---|---|---|---|---|---|\n"
        )
        for step in checkpoints:
            step_data = metrics.get(step, {})
            if band not in step_data:
                continue
            m = step_data[band]["metrics"]
            lines.append(
                f"| {step:02d} | {m['match_pct']:.2f}% | {m['mae']:.6f} | "
                f"{m['rmse']:.6f} | {m['max_err']:.6f} | {m['n_cells']:,} |\n"
            )
        lines.append("\n")

    return "".join(lines)


if __name__ == "__main__":
    run_cli(CoastalRasterValidationExecutor)
