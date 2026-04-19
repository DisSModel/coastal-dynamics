"""
tests/test_level1.py — Level 1: Analytical tests on minimal grids
==================================================================

Each test builds a 3×3 grid with fully controlled initial state,
computes the expected result by hand, and asserts that both the
Raster and Vector implementations produce the same output.

Grid layout (row, col):
    (0,0) (0,1) (0,2)
    (1,0) (1,1) (1,2)
    (2,0) (2,1) (2,2)

Center cell = index (1,1) = GDF row 4 (row-major order).

Run with:
    pytest tests/test_level1.py -v
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import geopandas as gpd
import pytest
from shapely.geometry import box

from dissmodel.core import Environment
from dissmodel.geo.raster.backend import RasterBackend

from coastal_dynamics.raster.flood_model    import FloodModel    as RasterFlood
from coastal_dynamics.raster.mangrove_model import MangroveModel as RasterMangue
from coastal_dynamics.vector.flood_model    import FloodModel    as VectorFlood
from coastal_dynamics.vector.mangrove_model import MangroveModel as VectorMangue

from coastal_dynamics.common.constants import (
    MAR, MANGUE, MANGUE_MIGRADO, MANGUE_INUNDADO,
    VEGETACAO_TERRESTRE,  VEG_TERRESTRE_INUNDADA,  # noqa: F401 — kept for clarity
    SOLO_DESCOBERTO, SOLO_INUNDADO,
    AREA_ANTROPIZADA, AREA_ANTROPIZADA_INUNDADA,
    SOLO_MANGUE, SOLO_MANGUE_MIGRADO, SOLO_CANAL_FLUVIAL, SOLO_OUTROS,
    VEG_TERRESTRE_INUNDADA,
)

# ── helpers ───────────────────────────────────────────────────────────────────

CELL_SIZE = 1.0   # 1 m cells — keeps coordinates simple


def make_gdf(uso: list[int], alt: list[float], solo: list[int]) -> gpd.GeoDataFrame:
    """
    Build a 3×3 GeoDataFrame from flat lists (row-major order).
    Each cell is a 1×1 square. Adds 'row' and 'col' integer indices.
    """
    assert len(uso) == len(alt) == len(solo) == 9, "Need exactly 9 values"
    rows, cols, geoms = [], [], []
    for r in range(3):
        for c in range(3):
            rows.append(r)
            cols.append(c)
            geoms.append(box(c, r, c + CELL_SIZE, r + CELL_SIZE))

    return gpd.GeoDataFrame(
        {"uso": uso, "alt": alt, "solo": solo, "row": rows, "col": cols},
        geometry=geoms,
        crs="EPSG:31984",
    )


def make_backend(uso: list[int], alt: list[float], solo: list[int]) -> RasterBackend:
    """Build a 3×3 RasterBackend from flat lists (row-major order)."""
    assert len(uso) == len(alt) == len(solo) == 9
    shape = (3, 3)
    backend = RasterBackend(shape=shape)
    backend.set("uso",  np.array(uso,  dtype=np.int16).reshape(shape))
    backend.set("alt",  np.array(alt,  dtype=np.float32).reshape(shape))
    backend.set("solo", np.array(solo, dtype=np.int16).reshape(shape))
    mask = np.ones(shape, dtype=bool)
    backend.set("mask", mask)
    return backend


def run_vector(gdf: gpd.GeoDataFrame, model_cls, n_steps: int = 1, **kwargs):
    """Run a single vector model for n_steps and return the mutated GDF."""
    env = Environment(start_time=1, end_time=n_steps)
    model_cls(gdf=gdf, **kwargs)
    env.run()
    return gdf


def run_raster(backend: RasterBackend, model_cls, n_steps: int = 1, **kwargs):
    """Run a single raster model for n_steps and return the backend."""
    env = Environment(start_time=1, end_time=n_steps)
    model_cls(backend=backend, **kwargs)
    env.run()
    return backend


# ── FloodModel tests ──────────────────────────────────────────────────────────

class TestFloodModel:
    """
    Analytical tests for FloodModel (raster and vector).

    All expected values are derived from the hidro.lua rules:
      1. Elevation diffusion (relative condition — flow shared between
         source and low neighbors)
      2. Flooding (absolute condition — alt <= nivel_mar)
    """

    def test_no_sea_cell_no_change(self):
        """
        If no cell is MAR or flooded, nothing should change.

        Expected: all uso and alt values remain identical after 1 step.
        """
        uso  = [VEGETACAO_TERRESTRE] * 9
        alt  = [5.0] * 9
        solo = [SOLO_OUTROS] * 9

        gdf     = make_gdf(uso, alt, solo)
        backend = make_backend(uso, alt, solo)

        run_vector(gdf,     VectorFlood, taxa_elevacao=0.011)
        run_raster(backend, RasterFlood, taxa_elevacao=0.011)

        assert (gdf["uso"] == VEGETACAO_TERRESTRE).all(), \
            "Vector: uso should not change without a sea source"
        assert (backend.get("uso") == VEGETACAO_TERRESTRE).all(), \
            "Raster: uso should not change without a sea source"

    def test_sea_cell_floods_low_neighbor(self):
        """
        Center (1,1): MAR, alt=0.
        All 8 neighbors: VEGETACAO_TERRESTRE, alt=0.005.

        nivel_mar = 1 * 0.011 = 0.011
        Neighbors (0.005) <= nivel_mar → all 8 flood: VEG_TERRESTRE_INUNDADA

        Elevation diffusion:
        neighbors alt (0.005) > center alt (0.0) → NOT lower than source
        viz_baixos = 1 (only center itself)
        fluxo = 0.011 / 1 = 0.011 → goes entirely to center
        neighbor alt: unchanged (0.005)
        """
        uso  = [VEGETACAO_TERRESTRE] * 9
        alt  = [0.005] * 9
        solo = [SOLO_OUTROS] * 9
        uso[4] = MAR
        alt[4] = 0.0

        gdf     = make_gdf(uso, alt, solo)
        backend = make_backend(uso, alt, solo)

        run_vector(gdf,     VectorFlood, taxa_elevacao=0.011)
        run_raster(backend, RasterFlood, taxa_elevacao=0.011)

        expected_uso = [VEG_TERRESTRE_INUNDADA] * 9
        expected_uso[4] = MAR

        assert list(gdf["uso"]) == expected_uso, \
            f"Vector uso mismatch: {list(gdf['uso'])} != {expected_uso}"
        assert list(backend.get("uso").flatten()) == expected_uso, \
            f"Raster uso mismatch: {list(backend.get('uso').flatten())} != {expected_uso}"

        # center absorbs full flux; neighbors are higher so receive nothing
        assert gdf.loc[4, "alt"] == pytest.approx(0.011, abs=1e-5), \
            "Vector: center alt wrong"
        assert backend.get("alt")[1, 1] == pytest.approx(0.011, abs=1e-5), \
            "Raster: center alt wrong"

        for i in [0, 1, 2, 3, 5, 6, 7, 8]:
            assert gdf.iloc[i]["alt"] == pytest.approx(0.005, abs=1e-5), \
                f"Vector: neighbor {i} alt should be unchanged"
        assert np.allclose(
            backend.get("alt").flatten()[[0,1,2,3,5,6,7,8]],
            0.005, atol=1e-5
        ), "Raster: neighbor alt should be unchanged"


    def test_flux_spreads_to_lower_neighbors(self):
        """
        Tests elevation diffusion when neighbors ARE lower than the source.

        Center (1,1): MAR, alt=0.1.
        All 8 neighbors: VEGETACAO_TERRESTRE, alt=0.0.

        nivel_mar = 0.011 — neighbors (0.0) <= nivel_mar → flood
        Diffusion: neighbors (0.0) <= center (0.1) → viz_baixos = 9
        fluxo = 0.011 / 9 ≈ 0.001222
        center alt:   0.1   + 0.001222 = 0.101222
        neighbor alt: 0.0   + 0.001222 = 0.001222
        """
        uso  = [VEGETACAO_TERRESTRE] * 9
        alt  = [0.0] * 9
        solo = [SOLO_OUTROS] * 9
        uso[4] = MAR
        alt[4] = 0.1

        gdf     = make_gdf(uso, alt, solo)
        backend = make_backend(uso, alt, solo)

        run_vector(gdf,     VectorFlood, taxa_elevacao=0.011)
        run_raster(backend, RasterFlood, taxa_elevacao=0.011)

        fluxo = 0.011 / 9

        assert gdf.loc[4, "alt"] == pytest.approx(0.1 + fluxo, abs=1e-5), \
            "Vector: center alt wrong"
        assert backend.get("alt")[1, 1] == pytest.approx(0.1 + fluxo, abs=1e-5), \
            "Raster: center alt wrong"

        for i in [0, 1, 2, 3, 5, 6, 7, 8]:
            assert gdf.iloc[i]["alt"] == pytest.approx(fluxo, abs=1e-5), \
                f"Vector: neighbor {i} alt wrong"
        assert np.allclose(
            backend.get("alt").flatten()[[0,1,2,3,5,6,7,8]],
            fluxo, atol=1e-5
        ), "Raster: neighbor alt wrong"

    def test_high_neighbor_not_flooded(self):
        """
        Center (1,1): MAR, alt=0.
        All neighbors: VEGETACAO_TERRESTRE, alt=10.0.

        nivel_mar = 0.011 — neighbors (10.0) >> nivel_mar → no flooding.
        Elevation: neighbors are NOT lower than center (10 > 0),
        so viz_baixos = 1 (only center itself).
        fluxo = 0.011 / 1 = 0.011 → goes entirely to center.
        No neighbor gets extra altitude.
        """
        uso  = [VEGETACAO_TERRESTRE] * 9
        alt  = [10.0] * 9
        solo = [SOLO_OUTROS] * 9
        uso[4] = MAR
        alt[4] = 0.0

        gdf     = make_gdf(uso, alt, solo)
        backend = make_backend(uso, alt, solo)

        run_vector(gdf,     VectorFlood, taxa_elevacao=0.011)
        run_raster(backend, RasterFlood, taxa_elevacao=0.011)

        # uso: no neighbor should be flooded
        for i in [0, 1, 2, 3, 5, 6, 7, 8]:
            assert gdf.iloc[i]["uso"] == VEGETACAO_TERRESTRE, \
                f"Vector: neighbor {i} should not be flooded"
        assert (backend.get("uso")[[0,1,2],[0,1,2]] == VEGETACAO_TERRESTRE).all() or \
               (backend.get("uso").flatten()[[0,1,2,3,5,6,7,8]] == VEGETACAO_TERRESTRE).all(), \
               "Raster: no neighbor should be flooded"

        # elevation: center absorbs all flow, neighbors unchanged
        assert gdf.loc[4, "alt"] == pytest.approx(0.011, abs=1e-5), \
            "Vector: center alt should be 0.011"
        assert backend.get("alt")[1, 1] == pytest.approx(0.011, abs=1e-5), \
            "Raster: center alt should be 0.011"

    def test_raster_vector_equivalence_random(self):
        """
        Smoke test: raster and vector produce identical uso after 1 step
        on a random but deterministic 3×3 grid.
        """
        rng  = np.random.default_rng(42)
        uso  = rng.choice([MAR, VEGETACAO_TERRESTRE, AREA_ANTROPIZADA], size=9).tolist()
        alt  = rng.uniform(0.0, 0.02, size=9).tolist()
        solo = [SOLO_OUTROS] * 9

        gdf     = make_gdf(list(uso), list(alt), list(solo))
        backend = make_backend(list(uso), list(alt), list(solo))

        run_vector(gdf,     VectorFlood, taxa_elevacao=0.011)
        run_raster(backend, RasterFlood, taxa_elevacao=0.011)

        vec_uso = list(gdf["uso"])
        ras_uso = list(backend.get("uso").flatten())

        assert vec_uso == ras_uso, \
            f"Raster/Vector uso diverged:\n  vector={vec_uso}\n  raster={ras_uso}"


# ── MangroveModel tests ───────────────────────────────────────────────────────

class TestMangroveModel:
    """
    Analytical tests for MangroveModel (raster and vector).

    Rules from mangue.lua:
      migrateSoils: source soil (MANGUE/MIGRADO/CANAL) → neighbor with
                    TARGET_USE and alt <= zi becomes SOLO_MANGUE_MIGRADO
      migrateUses:  source use (MANGUE/MIGRADO) → neighbor with
                    TARGET_USE + MANGROVE_SOIL + alt <= zi becomes MANGUE_MIGRADO
    """

    def test_no_source_no_migration(self):
        """
        If no cell has a source soil or source use, nothing migrates.
        """
        uso  = [VEGETACAO_TERRESTRE] * 9
        alt  = [1.0] * 9
        solo = [SOLO_OUTROS] * 9

        gdf     = make_gdf(uso, alt, solo)
        backend = make_backend(uso, alt, solo)

        run_vector(gdf,     VectorMangue, taxa_elevacao=0.011, altura_mare=6.0)
        run_raster(backend, RasterMangue, taxa_elevacao=0.011, altura_mare=6.0)

        assert (gdf["uso"]  == VEGETACAO_TERRESTRE).all(), "Vector: uso should not change"
        assert (gdf["solo"] == SOLO_OUTROS).all(),         "Vector: solo should not change"
        assert (backend.get("uso")  == VEGETACAO_TERRESTRE).all(), "Raster: uso unchanged"
        assert (backend.get("solo") == SOLO_OUTROS).all(),         "Raster: solo unchanged"

    def test_soil_migration_triggered(self):
        """
        Center (1,1): MANGUE, SOLO_MANGUE, alt=1.0 (source cell).
        All neighbors: VEGETACAO_TERRESTRE, SOLO_OUTROS, alt=1.0.

        zi = 6.0 + 1*0.011 = 6.011
        alt (1.0) <= zi (6.011) → condition met
        neighbor solo != SOLO_MANGUE_MIGRADO → condition met
        neighbor uso in TARGET_USES → condition met

        Expected: all 8 neighbors → solo = SOLO_MANGUE_MIGRADO
        uso of neighbors stays VEGETACAO_TERRESTRE (migrateUses requires
        neighbor solo in MANGROVE_SOILS — which was SOLO_OUTROS in solo_past)
        """
        uso  = [VEGETACAO_TERRESTRE] * 9
        alt  = [1.0] * 9
        solo = [SOLO_OUTROS] * 9
        uso[4]  = MANGUE
        solo[4] = SOLO_MANGUE

        gdf     = make_gdf(uso, alt, solo)
        backend = make_backend(uso, alt, solo)

        run_vector(gdf,     VectorMangue, taxa_elevacao=0.011, altura_mare=6.0)
        run_raster(backend, RasterMangue, taxa_elevacao=0.011, altura_mare=6.0)

        # solo: all neighbors become SOLO_MANGUE_MIGRADO; center unchanged
        expected_solo = [SOLO_MANGUE_MIGRADO] * 9
        expected_solo[4] = SOLO_MANGUE

        assert list(gdf["solo"]) == expected_solo, \
            f"Vector solo: {list(gdf['solo'])} != {expected_solo}"
        assert list(backend.get("solo").flatten()) == expected_solo, \
            f"Raster solo: {list(backend.get('solo').flatten())} != {expected_solo}"

        # uso: neighbors stay VEGETACAO_TERRESTRE (solo_past was SOLO_OUTROS)
        for i in [0,1,2,3,5,6,7,8]:
            assert gdf.iloc[i]["uso"] == VEGETACAO_TERRESTRE, \
                f"Vector: neighbor {i} uso should not migrate yet (solo_past=OUTROS)"
        assert (backend.get("uso").flatten()[[0,1,2,3,5,6,7,8]] == VEGETACAO_TERRESTRE).all(), \
            "Raster: neighbor uso should not migrate yet (solo_past=OUTROS)"

    def test_use_migration_requires_mangrove_soil(self):
        """
        Center: MANGUE, SOLO_MANGUE, alt=1.0.
        Neighbors: VEGETACAO_TERRESTRE, SOLO_MANGUE (already mangrove soil), alt=1.0.

        In this case solo_past IS in MANGROVE_SOILS → migrateUses triggers.
        Expected: all 8 neighbors → uso = MANGUE_MIGRADO
        """
        uso  = [VEGETACAO_TERRESTRE] * 9
        alt  = [1.0] * 9
        solo = [SOLO_MANGUE] * 9    # all cells already have mangrove soil
        uso[4] = MANGUE

        gdf     = make_gdf(uso, alt, solo)
        backend = make_backend(uso, alt, solo)

        run_vector(gdf,     VectorMangue, taxa_elevacao=0.011, altura_mare=6.0)
        run_raster(backend, RasterMangue, taxa_elevacao=0.011, altura_mare=6.0)

        expected_uso = [MANGUE_MIGRADO] * 9
        expected_uso[4] = MANGUE

        assert list(gdf["uso"]) == expected_uso, \
            f"Vector uso: {list(gdf['uso'])} != {expected_uso}"
        assert list(backend.get("uso").flatten()) == expected_uso, \
            f"Raster uso: {list(backend.get('uso').flatten())} != {expected_uso}"

    def test_high_altitude_blocks_migration(self):
        """
        Center: MANGUE, SOLO_MANGUE, alt=1.0.
        Neighbors: VEGETACAO_TERRESTRE, SOLO_MANGUE, alt=100.0.

        zi = 6.011. Neighbor alt (100.0) > zi → migration blocked.
        Expected: nothing changes.
        """
        uso  = [VEGETACAO_TERRESTRE] * 9
        alt  = [100.0] * 9
        solo = [SOLO_MANGUE] * 9
        uso[4]  = MANGUE
        alt[4]  = 1.0

        gdf     = make_gdf(uso, alt, solo)
        backend = make_backend(uso, alt, solo)

        run_vector(gdf,     VectorMangue, taxa_elevacao=0.011, altura_mare=6.0)
        run_raster(backend, RasterMangue, taxa_elevacao=0.011, altura_mare=6.0)

        for i in [0,1,2,3,5,6,7,8]:
            assert gdf.iloc[i]["uso"] == VEGETACAO_TERRESTRE, \
                f"Vector: neighbor {i} uso should not migrate (alt too high)"
        assert (backend.get("uso").flatten()[[0,1,2,3,5,6,7,8]] == VEGETACAO_TERRESTRE).all(), \
            "Raster: neighbor uso should not migrate (alt too high)"

    def test_raster_vector_equivalence_uso_and_solo(self):
        """
        Smoke test: raster and vector produce identical uso and solo
        after 3 steps on a deterministic mixed grid.
        """
        uso  = [VEGETACAO_TERRESTRE, MANGUE,             VEGETACAO_TERRESTRE,
                SOLO_DESCOBERTO,     MANGUE_MIGRADO,     VEGETACAO_TERRESTRE,
                VEGETACAO_TERRESTRE, VEGETACAO_TERRESTRE, SOLO_DESCOBERTO]
        alt  = [1.0] * 9
        solo = [SOLO_MANGUE,    SOLO_MANGUE,         SOLO_MANGUE,
                SOLO_MANGUE,    SOLO_MANGUE,         SOLO_MANGUE_MIGRADO,
                SOLO_CANAL_FLUVIAL, SOLO_OUTROS,     SOLO_OUTROS]

        gdf     = make_gdf(list(uso), list(alt), list(solo))
        backend = make_backend(list(uso), list(alt), list(solo))

        run_vector(gdf,     VectorMangue, n_steps=3, taxa_elevacao=0.011, altura_mare=6.0)
        run_raster(backend, RasterMangue, n_steps=3, taxa_elevacao=0.011, altura_mare=6.0)

        vec_uso  = list(gdf["uso"])
        ras_uso  = list(backend.get("uso").flatten())
        vec_solo = list(gdf["solo"])
        ras_solo = list(backend.get("solo").flatten())

        assert vec_uso == ras_uso, \
            f"Raster/Vector uso diverged after 3 steps:\n  vec={vec_uso}\n  ras={ras_uso}"
        assert vec_solo == ras_solo, \
            f"Raster/Vector solo diverged after 3 steps:\n  vec={vec_solo}\n  ras={ras_solo}"


# ── cross-model invariants ────────────────────────────────────────────────────

class TestInvariants:
    """
    Properties that must hold by construction regardless of input,
    verified on both raster and vector simultaneously.
    """

    def test_flooded_cells_monotonically_nondecreasing(self):
        """
        Once a cell is flooded, it should not revert to a dry state
        (no accretion active, no drainage rule).
        """
        uso  = [VEGETACAO_TERRESTRE] * 9
        alt  = [0.005] * 9
        solo = [SOLO_OUTROS] * 9
        uso[4] = MAR
        alt[4] = 0.0

        from coastal_dynamics.common.constants import USOS_INUNDADOS

        gdf     = make_gdf(uso, alt, solo)
        backend = make_backend(uso, alt, solo)

        n_steps = 5
        env_vec = Environment(start_time=1, end_time=n_steps)
        env_ras = Environment(start_time=1, end_time=n_steps)

        flood_vec = VectorFlood(gdf=gdf, taxa_elevacao=0.011)
        flood_ras = RasterFlood(backend=backend, taxa_elevacao=0.011)

        # track flooded count each step
        flooded_vec_counts = []
        flooded_ras_counts = []

        original_vec_execute = flood_vec.execute
        original_ras_execute = flood_ras.execute

        def patched_vec():
            original_vec_execute()
            flooded_vec_counts.append(int(gdf["uso"].isin(USOS_INUNDADOS).sum()))

        def patched_ras():
            original_ras_execute()
            flooded_ras_counts.append(int(np.isin(backend.get("uso"), USOS_INUNDADOS).sum()))

        flood_vec.execute = patched_vec
        flood_ras.execute = patched_ras

        env_vec.run()
        env_ras.run()

        for i in range(1, len(flooded_vec_counts)):
            assert flooded_vec_counts[i] >= flooded_vec_counts[i - 1], \
                f"Vector: flooded count decreased at step {i+1}"
            assert flooded_ras_counts[i] >= flooded_ras_counts[i - 1], \
                f"Raster: flooded count decreased at step {i+1}"

    def test_uso_values_always_valid(self):
        """
        After any number of steps, uso values must remain within the
        set of valid land-use codes {1..10}.
        """
        valid_usos = set(range(1, 11))

        uso  = [VEGETACAO_TERRESTRE, MAR, MANGUE,
                AREA_ANTROPIZADA,   MAR,  SOLO_DESCOBERTO,
                MANGUE,             MAR,  VEGETACAO_TERRESTRE]
        alt  = [0.01] * 9
        solo = [SOLO_MANGUE] * 9

        gdf     = make_gdf(uso, alt, solo)
        backend = make_backend(uso, alt, solo)

        run_vector(gdf,     VectorFlood, n_steps=5, taxa_elevacao=0.011)
        run_raster(backend, RasterFlood, n_steps=5, taxa_elevacao=0.011)

        vec_invalid = set(gdf["uso"]) - valid_usos
        ras_invalid = set(backend.get("uso").flatten().tolist()) - valid_usos

        assert not vec_invalid, f"Vector: invalid uso values found: {vec_invalid}"
        assert not ras_invalid, f"Raster: invalid uso values found: {ras_invalid}"
