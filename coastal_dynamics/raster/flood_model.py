"""
flood_raster_model.py — Hydrological Model for DisSModel
========================================================
Faithful translation of hidro.lua to DisSModel + RasterBackend.
"""
from __future__ import annotations

import numpy as np
from dissmodel.geo.raster.backend import RasterBackend
from dissmodel.geo.raster.sync_model import SyncRasterModel

from coastal_dynamics.common.constants import (
    USOS_INUNDADOS,
    REGRAS_INUNDACAO,
    MAR,
)


class FloodModel(SyncRasterModel):
    """
    Hydrological model (hidro.lua) → DisSModel + RasterBackend.

    Uses shared snapshot semantics (auto_sync=False): the ``StepSyncModel``
    created in the executor freezes "uso" and "alt" in ``"uso_past"`` /
    ``"alt_past"`` before this model and MangroveModel run, ensuring both
    read the state at the beginning of the step — equivalent to TerraME's
    ``cell.past[attr]``.

    Parameters
    ----------
    backend       : RasterBackend containing arrays "uso" and "alt"
    taxa_elevacao : meters/year — IPCC RCP8.5 ≈ 0.011
    """

    def setup(
        self,
        backend:       RasterBackend,
        taxa_elevacao: float = 0.011,
    ) -> None:
        super().setup(backend)
        self.land_use_types    = ["uso", "alt"]


        self.taxa_elevacao     = taxa_elevacao
        self.flooded_cells     = 0
        self.newly_flooded     = 0
        self.current_sea_level = 0.0

    def execute(self) -> None:
        nivel_mar  = self.env.now() * self.taxa_elevacao
        rows, cols = self.shape

        # mask: True = valid cell (covered by a polygon)
        # falls back to all-True if backend was loaded from GeoTIFF (no mask band)
        mask = self.backend.arrays.get(
            "mask", np.ones((rows, cols), dtype=bool)
        ).astype(bool)

        # read shared snapshot frozen by StepSyncModel at step start
        # equivalent to TerraME's cell.past["uso"] / cell.past["alt"]
        uso_past = self.backend.get("uso_past")
        alt_past = self.backend.get("alt_past")

        # source cells: already flooded or sea — only within valid area
        eh_fonte = np.isin(uso_past, USOS_INUNDADOS) & (alt_past >= 0) & mask

        viz_baixos = np.ones((rows, cols), dtype=float)
        for dr, dc in self.dirs:
            viz_baixos += (self.shift(alt_past, dr, dc) <= alt_past).astype(float)

        fluxo     = np.where(eh_fonte, self.taxa_elevacao / viz_baixos, 0.0)
        delta_alt = fluxo.copy()
        uso_novo  = uso_past.copy()

        for dr, dc in self.dirs:
            fonte_viz = self.shift(eh_fonte.astype(float), dr, dc) > 0
            alt_viz   = self.shift(alt_past, dr, dc)
            fluxo_viz = self.shift(fluxo, dr, dc)

            # 1. elevation update — relative condition
            delta_alt += np.where(
                fonte_viz & (alt_past <= alt_viz), fluxo_viz, 0.0
            )

            # 2. flooding — absolute elevation threshold
            for uso_seco, uso_inund in REGRAS_INUNDACAO.items():
                pode = (
                    fonte_viz
                    & (uso_past == uso_seco)
                    & (alt_past <= nivel_mar)
                    & mask          # never flood outside valid area
                )
                uso_novo = np.where(pode, uso_inund, uso_novo)

        # final guard: cells outside mask always keep their original values
        alt_novo = alt_past + delta_alt
        self.backend.arrays["alt"] = np.where(mask, alt_novo, alt_past)
        self.backend.arrays["uso"] = np.where(mask, uso_novo, uso_past)

        # metrics
        inund = np.isin(uso_novo, USOS_INUNDADOS) & (uso_novo != MAR) & mask
        novas = (
            np.isin(uso_novo, USOS_INUNDADOS)
            & ~np.isin(uso_past, USOS_INUNDADOS)
            & mask
        )

        self.flooded_cells     = int(np.sum(inund))
        self.newly_flooded     = int(np.sum(novas))
        self.current_sea_level = round(nivel_mar, 4)