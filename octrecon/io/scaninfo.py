"""ScanInfo.json parsing and tile bookkeeping for a tiled OCT volume.

Mirrors the fields used by yOCTProcessTiledScan.m / _createDimStructure.m /
_getScansFromYFrame.m of the legacy myOCT library.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Tile:
    folder: str        # e.g. 'Data01'
    xi: int            # index into x_centers_mm
    yi: int            # index into y_centers_mm
    zi: int            # index into z_depths_mm
    x_center_mm: float
    y_center_mm: float
    z_depth_mm: float


class ScanInfo:
    def __init__(self, volume_folder: str | Path):
        self.folder = Path(volume_folder)
        with open(self.folder / "ScanInfo.json", "r") as f:
            self.json = json.load(f)
        j = self.json

        # yOCTProcessTiledScan.m l.141-148: octSystem (or deprecated OCTSystem) field.
        self.oct_system = j.get("octSystem", j.get("OCTSystem"))
        self.oct_system_source = "ScanInfo.json octSystem" if "octSystem" in j else "ScanInfo.json OCTSystem"
        self.z_depths_mm = np.asarray(j["zDepths"], dtype=float).ravel()
        self.x_centers_mm = np.asarray(j.get("xCenters_mm", j.get("xCenters")), dtype=float).ravel()
        self.y_centers_mm = np.asarray(j.get("yCenters_mm", j.get("yCenters")), dtype=float).ravel()
        self.tissue_ri = float(j["tissueRefractiveIndex"])
        self.probe = j["octProbe"]
        self.n_x_px = int(j["nXPixelsInEachTile"])
        self.n_y_px = int(j["nYPixelsInEachTile"])
        self.tile_range_x_mm = float(j["tileRangeX_mm"])
        self.tile_range_y_mm = float(j["tileRangeY_mm"])
        self.x_offset = float(j.get("xOffset", 0))
        self.y_offset = float(j.get("yOffset", 0))
        self.pixel_size_um = float(j.get("pixelSize_um", np.nan))
        self.n_bscan_avg = int(j.get("nBScanAvg", 1))

        gx = np.asarray(j["gridXcc"], float).ravel()
        gy = np.asarray(j["gridYcc"], float).ravel()
        gz = np.asarray(j["gridZcc"], float).ravel()
        folders = j["octFolders"]
        if isinstance(folders, str):
            folders = [folders]
        tiles = []
        for f, x, y, z in zip(folders, gx, gy, gz):
            xi = int(np.argmin(np.abs(self.x_centers_mm - x)))
            yi = int(np.argmin(np.abs(self.y_centers_mm - y)))
            zi = int(np.argmin(np.abs(self.z_depths_mm - z)))
            assert abs(self.x_centers_mm[xi] - x) < 1e-9
            assert abs(self.y_centers_mm[yi] - y) < 1e-9
            assert abs(self.z_depths_mm[zi] - z) < 1e-9
            tiles.append(Tile(f, xi, yi, zi, x, y, z))
        self.tiles = tiles

        if not self.oct_system:
            # Legacy yOCTProcessTiledScan errors here; we fall back to the detection
            # yOCTLoadInterfFromFile uses for single scans (WhatOCTSystemIsIt) on tile 1.
            from .detect import what_oct_system_is_it
            self.oct_system, man = what_oct_system_is_it(self.folder / tiles[0].folder)
            self.oct_system_source = f"auto-detected from {tiles[0].folder} ({man})"

    def optical_path_polynomial(self):
        p = self.probe.get("OpticalPathCorrectionPolynomial")
        return None if p is None else np.asarray(p, float).ravel()

    def tiles_in_row(self, yi: int) -> list[Tile]:
        """All tiles of one y-tile row, ordered x-major / z-minor exactly like
        yOCTProcessTiledScan_getScansFromYFrame (octFolders order filtered by y)."""
        row = [t for t in self.tiles if t.yi == yi]
        # legacy loop assumes (xxI, zzI) nested order; verify it holds
        expect = [(xi, zi) for xi in range(len(self.x_centers_mm)) for zi in range(len(self.z_depths_mm))]
        got = [(t.xi, t.zi) for t in row]
        if got != expect:
            raise ValueError(f"Tile order in y-row {yi} differs from legacy (x-major, z-minor) assumption")
        return row
