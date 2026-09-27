"""Optical-path (field curvature) correction + focus-weighted tile stitching.

Equivalent MATLAB:
  yOCTOpticalPathCorrection.m      nearest-neighbour z resampling at z + P(x, y)
  yOCTProcessTiledScan_factorZ.m   focus weight exp(-(z-f)^2/(2 sigma)^2) + exp(-4.5)
  yOCTProcessTiledScan.m 300-371   interp2(linear, extrap 0) into output grid,
                                   weighted mean, NaN where weight < exp(-4.5)

The legacy code interp2's every B-scan onto the full-width output plane
(35 x 6000). Here interpolation is expressed as two small separable linear
operators (Wz: nOutZ x nRows, Wx: nTileX x nCols) restricted to the rows /
columns that can actually receive data.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

CUTOFF_SIGMA = 3.0
MIN_WEIGHT = float(np.exp(-CUTOFF_SIGMA ** 2 / 2))


def min_weight_threshold(dtype) -> float:
    """Legacy sets weight < exp(-4.5) to NaN. A pixel covered by a single tile at its floor
    weight sits *exactly* on the threshold (kept by legacy); allow a few ulps of the
    accumulation dtype so float32 rounding cannot flip it to NaN."""
    return MIN_WEIGHT * (1.0 - 8 * float(np.finfo(dtype).eps))


def factor_z(n_rows: int, focus_pix_1based: float, focus_sigma: float) -> np.ndarray:
    zI = np.arange(1, n_rows + 1, dtype=float)
    return np.exp(-(zI - focus_pix_1based) ** 2 / (2 * focus_sigma) ** 2) + np.exp(-3.0 ** 2 / 2)


def linear_interp_matrix(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """W (len(dst), len(src)) such that W @ v == interp1(src, v, dst, 'linear', 0)
    for strictly increasing src (MATLAB griddedInterpolant semantics)."""
    W = np.zeros((len(dst), len(src)))
    inside = (dst >= src[0]) & (dst <= src[-1])
    q = dst[inside]
    j = np.searchsorted(src, q, side="right") - 1
    j = np.clip(j, 0, len(src) - 2)
    t = (q - src[j]) / (src[j + 1] - src[j])
    # queries on a grid node give exactly the node value (as MATLAB interp2 does)
    t[np.abs(t) < 1e-9] = 0.0
    t[np.abs(1.0 - t) < 1e-9] = 1.0
    rows = np.nonzero(inside)[0]
    W[rows, j] += 1 - t
    W[rows, j + 1] += t
    return W


@dataclass
class OpticalPathCorrection:
    """Per (frame, column) integer row shift + validity bounds.

    Legacy: out(r, c) = in(nearest(z_r + P(x_c, y)), c), NaN (-> 0, invalid)
    when z_r + P is outside [z_first, z_last].  Because the tile z grid is
    uniform, nearest(z_r + P) = r + round(P/dz) and the validity test is
    0 <= r + P/dz <= nZ-1.  Shift/bounds are identical for every tile.
    """
    shift: np.ndarray    # (nY, nX) int   row offset
    pdz: np.ndarray      # (nY, nX) float P/dz (for validity)
    n_rows: int

    @classmethod
    def build(cls, poly, tile_x_mm, tile_y_mm, tile_z_mm):
        x = tile_x_mm * 1e3
        y = tile_y_mm * 1e3
        z = tile_z_mm * 1e3
        dz = (z[-1] - z[0]) / (len(z) - 1)
        X, Y = np.meshgrid(x, y)                       # (nY, nX)
        P = poly[0] * X + poly[1] * Y + poly[2] * X ** 2 + poly[3] * Y ** 2 + poly[4] * X * Y
        pdz = P / dz
        # nearest index of z_r + P on the uniform grid, relative to r
        shift = np.floor(pdz + 0.5).astype(np.int64)
        return cls(shift, pdz, len(z))


class TileStitcher:
    """Precomputed operators for one tile position (xi, zi) of the grid."""

    def __init__(self, tg, oc: OpticalPathCorrection | None, xi_center_mm, z_depth_mm, focus_pix,
                 focus_sigma, xp=np, dtype=np.float32):
        self.xp = xp
        nz = len(tg.tile_z_mm)
        # tile position in the output frame (yOCTProcessTiledScan.m 315-331)
        x = tg.tile_x_mm + xi_center_mm
        z = tg.tile_z_mm + z_depth_mm
        if not np.isnan(focus_pix):
            z = z - tg.tile_z_mm[int(np.round(focus_pix)) - 1]
        x = x.copy(); z = z.copy()
        x[0] -= 1e-10; x[-1] += 1e-10
        z[0] -= 1e-10; z[-1] += 1e-10

        Wz = linear_interp_matrix(z, tg.out_z_mm)                  # (nOutZ, nZ)
        Wx = linear_interp_matrix(x, tg.out_x_mm)                  # (nOutX, nTileX)
        rows = np.nonzero(Wz.any(axis=0))[0]
        cols = np.nonzero(Wx.any(axis=1))[0]
        self.empty = rows.size == 0 or cols.size == 0
        if self.empty:
            return
        r0, r1 = rows[0], rows[-1] + 1
        c0, c1 = cols[0], cols[-1] + 1
        self.r0, self.r1, self.c0, self.c1 = r0, r1, c0, c1
        self.Wz = xp.asarray(Wz[:, r0:r1].astype(dtype))            # (nOutZ, R)
        self.WxT = xp.asarray(Wx[c0:c1, :].T.astype(dtype))         # (nTileX, C)

        fz = factor_z(nz, focus_pix, focus_sigma) if not np.isnan(focus_pix) else np.ones(nz)
        self.fz = xp.asarray(fz[r0:r1].astype(dtype))               # (R,)

        # optical path correction: shift map shared by all tiles, gathered per batch
        self.n_rows = nz
        self.rr = xp.arange(r0, r1, dtype=xp.int64)
        if oc is not None:
            self.shift = xp.asarray(oc.shift)                         # (nY, nX) int64
            self.pdz = xp.asarray(oc.pdz)                             # (nY, nX) float64
        else:
            self.shift = None

    def contribute(self, mag, frame_idx):
        """mag: (B, nTileX, nZ) magnitude of frames `frame_idx` (tile-local y).
        Returns (num, den): each (B, nOutZ, C) to add into output columns c0:c1."""
        xp = self.xp
        if self.shift is not None:
            src = self.rr + self.shift[frame_idx][..., None]            # (B, nX, R)
            t = self.rr + self.pdz[frame_idx][..., None]
            valid = (t >= 0) & (t <= self.n_rows - 1)
            src = xp.clip(src, 0, self.n_rows - 1)
        else:
            src = xp.broadcast_to(self.rr, (len(frame_idx), mag.shape[1], len(self.rr)))
            valid = xp.ones(src.shape, dtype=bool)
        a = xp.take_along_axis(mag, src, axis=-1)                   # OPC gather
        f = valid * self.fz                                         # (B, nX, R)
        a = a * f                                                   # invalid -> 0 (scan & factor)
        # (B, nX, R) @ Wz^T -> (B, nX, nOutZ); then contract x: -> (B, nOutZ, C)
        num = xp.matmul(xp.matmul(a, self.Wz.T).transpose(0, 2, 1), self.WxT)
        den = xp.matmul(xp.matmul(f, self.Wz.T).transpose(0, 2, 1), self.WxT)
        return num, den
