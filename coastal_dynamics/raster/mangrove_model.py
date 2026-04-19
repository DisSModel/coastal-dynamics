"""
mangrove_raster_model.py — Mangrove Model for DisSModel
=======================================================
Faithful translation of mangue.lua to DisSModel + RasterBackend.
"""
from __future__ import annotations

import numpy as np
from dissmodel.geo.raster.backend import RasterBackend
from dissmodel.geo.raster.sync_model import SyncRasterModel

from coastal_dynamics.common.constants import (
    MANGUE,
    MANGUE_MIGRADO,
    VEGETACAO_TERRESTRE,
    SOLO_DESCOBERTO,
    USOS_INUNDADOS,
    SOLO_MANGUE,
    SOLO_MANGUE_MIGRADO,
    SOLO_CANAL_FLUVIAL,
)


class MangroveModel(SyncRasterModel):
    """
    Mangrove model (mangue.lua) → DisSModel + RasterBackend.

    Uses shared snapshot semantics (auto_sync=False): the ``StepSyncModel``
    created in the executor freezes "uso", "alt" and "solo" in their
    ``_past`` counterparts before this model and FloodModel run, ensuring
    both read the state at the beginning of the step — equivalent to
    TerraME's ``cell.past[attr]``.

    Parameters
    ----------
    backend       : RasterBackend containing arrays "uso", "alt", and "solo"
    taxa_elevacao : meters/year — IPCC RCP8.5 ≈ 0.011
    altura_mare   : base tidal influence height (AIM) in meters. Default: 6.0
    acrecao_ativa : enables sediment accretion (Alongi 2008). Default: False
    """

    SOIL_SOURCES   = [SOLO_MANGUE, SOLO_MANGUE_MIGRADO, SOLO_CANAL_FLUVIAL]
    MANGROVE_SOILS = [SOLO_MANGUE, SOLO_MANGUE_MIGRADO]
    USE_SOURCES    = [MANGUE, MANGUE_MIGRADO]
    USE_TARGETS    = [VEGETACAO_TERRESTRE, SOLO_DESCOBERTO]
    COEF_A, COEF_B = 1.693, 0.939  # Alongi 2008

    def setup(
        self,
        backend:       RasterBackend,
        taxa_elevacao: float = 0.011,
        altura_mare:   float = 6.0,
        acrecao_ativa: bool  = False,
    ) -> None:
        super().setup(backend)
        self.land_use_types    = ["uso", "alt", "solo"]
        self.auto_sync         = False   # StepSyncModel handles synchronization

        self.taxa_elevacao     = taxa_elevacao
        self.altura_mare       = altura_mare
        self.acrecao_ativa     = acrecao_ativa
        self.mangrove_migrated = 0
        self.soil_migrated     = 0

    def execute(self) -> None:
        nivel_mar  = self.env.now() * self.taxa_elevacao
        zi         = self.altura_mare + nivel_mar
        taxa_ac    = self.COEF_A / 1000.0 + self.COEF_B * nivel_mar
        rows, cols = self.shape

        # mask: True = valid cell — falls back to all-True for GeoTIFF input
        mask = self.backend.arrays.get(
            "mask", np.ones((rows, cols), dtype=bool)
        ).astype(bool)

        # read shared snapshot frozen by StepSyncModel at step start
        # equivalent to TerraME's cell.past["uso"] / cell.past["alt"] / cell.past["solo"]
        uso_past  = self.backend.get("uso_past")
        alt_past  = self.backend.get("alt_past")
        solo_past = self.backend.get("solo_past")

        # ── soil migration ───────────────────────────────────────────────────
        eh_fonte_solo = np.isin(solo_past, self.SOIL_SOURCES) & mask
        solo_novo     = solo_past.copy()

        for dr, dc in self.dirs:
            fonte_viz = self.shift(eh_fonte_solo.astype(np.int8), dr, dc) > 0
            cond = (
                fonte_viz
                & np.isin(uso_past, self.USE_TARGETS)
                & (solo_past != SOLO_MANGUE_MIGRADO)
                & (alt_past <= zi)
                & mask
            )
            solo_novo = np.where(cond, SOLO_MANGUE_MIGRADO, solo_novo)

        # ── land-use migration — uses uso_past / solo_past (TerraME .past) ──
        eh_fonte_uso = np.isin(uso_past, self.USE_SOURCES) & mask
        uso_novo     = uso_past.copy()

        for dr, dc in self.dirs:
            fonte_viz = self.shift(eh_fonte_uso.astype(np.int8), dr, dc) > 0
            cond = (
                fonte_viz
                & np.isin(uso_past, self.USE_TARGETS)
                & np.isin(solo_past, self.MANGROVE_SOILS)
                & (alt_past <= zi)
                & mask
            )
            uso_novo = np.where(cond, MANGUE_MIGRADO, uso_novo)

        # ── sediment accretion (disabled by default) ─────────────────────────
        if self.acrecao_ativa:
            cond_ac = (
                np.isin(solo_past, self.MANGROVE_SOILS)
                & ~np.isin(uso_past, USOS_INUNDADOS)
                & mask
            )
            alt_novo = np.where(cond_ac, alt_past + taxa_ac, alt_past)
            self.backend.arrays["alt"] = np.where(mask, alt_novo, alt_past)

        # final guard: cells outside mask always keep their original values
        self.backend.arrays["uso"]  = np.where(mask, uso_novo,  uso_past)
        self.backend.arrays["solo"] = np.where(mask, solo_novo, solo_past)

        # metrics — only count valid cells
        self.mangrove_migrated = int(np.sum((uso_novo  == MANGUE_MIGRADO)      & mask))
        self.soil_migrated     = int(np.sum((solo_novo == SOLO_MANGUE_MIGRADO) & mask))