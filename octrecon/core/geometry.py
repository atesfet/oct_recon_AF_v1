"""Spectral and spatial geometry of a tiled scan (host-side, float64).

Equivalent MATLAB:
  yOCTLoadInterfFromFile_ThorlabsHeaderLambda.m  (chirp -> lambda)
  yOCTEquispaceInterf.m                          (k linearisation, sinc/pchip)
  yOCTInterfToScanCpx_getZ.m                     (depth axis)
  yOCTProcessTiledScan_createDimStructure.m      (tile / output grids)
  yOCTProcessTiledScan.m lines 171-222           (output z resampling + crop)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.interpolate import PchipInterpolator

# Wavelength range per system (nm), from yOCTLoadInterfFromFile_ThorlabsHeaderLambda.m
LAMBDA_RANGE_NM = {
    "ganymede": (796.23, 1010.02),
    "ganymede_srr": (796.23, 1010.02),
    "gan632": (796.0, 1010.0),   # legacy: "TODO these will need to be measured"
    "telesto": (1208.69, 1372.50),
    "telesto_srr": (1208.69, 1372.50),
}


# --------------------------------------------------------------------------
# MATLAB-compatible grid constructors (so grids agree with legacy to the ulp)
# --------------------------------------------------------------------------
def matlab_linspace(a: float, b: float, n: int) -> np.ndarray:
    n1 = n - 1
    y = a + (np.arange(n) * (b - a)) / n1
    y[0] = a
    y[-1] = b
    return y


def matlab_colon(a: float, d: float, b: float) -> np.ndarray:
    """a:d:b with MATLAB's symmetric evaluation (first half from a, second
    half from the end point) and its count tolerance."""
    tol = 2.0 * np.finfo(float).eps * max(abs(a), abs(b))
    n = int(np.floor((b - a) / d + tol / abs(d))) if d != 0 else -1
    if n < 0:
        return np.zeros(0)
    k = np.arange(n + 1, dtype=float)
    out = np.empty(n + 1)
    half = n // 2
    end = a + n * d
    out[: half + 1] = a + k[: half + 1] * d
    out[half + 1:] = end - (n - k[half + 1:]) * d
    return out


# --------------------------------------------------------------------------
# Spectral geometry
# --------------------------------------------------------------------------
def chirp_to_lambda(chirp: np.ndarray, oct_system: str) -> np.ndarray:
    lmin, lmax = LAMBDA_RANGE_NM[oct_system.lower()]
    n = len(chirp)
    return 1.0 / (chirp * (1.0 / (n - 1) * (1.0 / lmax - 1.0 / lmin)) + 1.0 / lmin)


def _pchip_matlab(x, v, xq):
    """interp1(x,v,xq,'pchip') for strictly monotonic x (either direction)."""
    x = np.asarray(x, float)
    if x[0] > x[-1]:
        x, v = x[::-1], v[::-1]
    return PchipInterpolator(x, v, extrapolate=True)(xq)


@dataclass
class SpectralGeometry:
    lambda_nm: np.ndarray        # raw per-pixel wavelength
    k: np.ndarray                # 2*pi/lambda (1/nm)
    k_lin: np.ndarray            # equispaced target k (descending, like legacy)
    lambda_eq_nm: np.ndarray     # wavelength of equispaced samples
    sinc_idx: np.ndarray         # (N, T) int   source sample indices (padded)
    sinc_w: np.ndarray           # (N, T) float sinc weights (0 for padding)
    window: np.ndarray           # (N,) complex: Hann/rms * exp(i*dispersion)
    z_um: np.ndarray             # (N/2,) depth in medium, z=0 at zero delay
    interp_method: str


def build_spectral_geometry(lambda_nm: np.ndarray, dispersion_quadratic_term: float,
                            n_medium: float, interp_method: str = "sinc5") -> SpectralGeometry:
    lam = np.asarray(lambda_nm, float).ravel()
    N = len(lam)
    k = 2 * np.pi / lam
    k_lin = matlab_linspace(k.max(), k.min(), N)

    m = interp_method.lower()
    if m.startswith("sinc"):
        taps = int(m[4:])
        nq = (k_lin - k_lin.min()) / (k_lin.max() - k_lin.min()) * (N - 1)
        n = (k - k.min()) / (k.max() - k.min()) * (N - 1)
        dist = nq[:, None] - n[None, :]                     # (N, N)
        mask = np.abs(dist) < taps
        T = int(mask.sum(1).max())
        idx = np.zeros((N, T), np.int64)
        w = np.zeros((N, T), float)
        for i in range(N):
            jj = np.nonzero(mask[i])[0]
            idx[i, : len(jj)] = jj
            w[i, : len(jj)] = np.sinc(dist[i, jj])
        # padding entries point at sample 0 with weight 0
    elif m == "pchip":
        raise NotImplementedError("pchip k-linearisation is not implemented in the GPU port "
                                  "(the 10um_FOV_1 run used sinc5)")
    else:
        raise ValueError(interp_method)

    lambda_eq = _pchip_matlab(k, lam, k_lin)             # legacy: 'pchip' for lambda
    k_eq = 2 * np.pi / lambda_eq

    # yOCTInterfToScanCpx: Hann window normalised by its RMS, quadratic dispersion
    hann = 0.5 * (1 - np.cos(2 * np.pi * np.arange(N) / (N - 1)))   # MATLAB hann(N) symmetric
    hann = hann / np.sqrt(np.mean(hann ** 2))
    phase = -dispersion_quadratic_term * (k_eq - k_eq.mean()) ** 2
    window = hann * np.exp(1j * phase)

    # yOCTInterfToScanCpx_getZ (uses first/last equispaced lambda)
    l_sorted = np.sort([lambda_eq[0], lambda_eq[-1]])
    lambda0_um = l_sorted.mean() / 1e3
    dlambda_um = np.diff(l_sorted)[0] / 1e3
    z_step = 0.5 * lambda0_um ** 2 / dlambda_um / n_medium
    z_um = matlab_linspace(0.0, z_step * N / 2, N // 2)

    return SpectralGeometry(lam, k, k_lin, lambda_eq, idx, w, window, z_um, interp_method)


# --------------------------------------------------------------------------
# Spatial geometry
# --------------------------------------------------------------------------
@dataclass
class TiledGeometry:
    tile_x_mm: np.ndarray      # (nX,) tile-local x
    tile_y_mm: np.ndarray      # (nY,) tile-local y
    tile_z_mm: np.ndarray      # (nZ,) tile-local z, 0 at reference focus pixel
    out_x_mm: np.ndarray
    out_y_mm: np.ndarray
    out_z_mm: np.ndarray       # after resampling + crop
    out_z_full_mm: np.ndarray  # native-dz output grid before resampling (for metadata)
    focus_pix: np.ndarray      # (nDepths,) 1-based focus pixel per zDepth (NaN = no gating)


def build_tiled_geometry(si, z_um_tile: np.ndarray, focus_pix, output_pixel_size_um: float | None,
                         crop_z_range_mm=None) -> TiledGeometry:
    nd = len(si.z_depths_mm)
    focus = np.asarray(focus_pix, float).ravel()
    if focus.size == 1:
        focus = np.full(nd, focus[0])
    if focus.size != nd:
        raise ValueError("focus_pix must be scalar or one value per zDepth")

    tx = si.x_offset + si.tile_range_x_mm * matlab_linspace(-0.5, 0.5, si.n_x_px + 1)
    ty = si.y_offset + si.tile_range_y_mm * matlab_linspace(-0.5, 0.5, si.n_y_px + 1)
    tx, ty = tx[:-1], ty[:-1]

    j = si.json
    if "galvoPhaseDelayXOffsetCorrection_mm" not in j:
        gpd = float(si.probe.get("GalvoPhaseDelay_Asamples", 0))
        if gpd != 0 and "pixelSize_um" in j:
            tx = tx - gpd * (j["pixelSize_um"] - 1.0) * 1e-3

    tz = z_um_tile / 1e3
    if not np.any(np.isnan(focus)):
        if np.all(focus == focus[0]):
            ref = focus[0]
        else:
            ref = focus[int(np.argmin(np.abs(si.z_depths_mm)))]
        tz = tz - tz[int(np.round(ref)) - 1]

    dx = tx[1] - tx[0]
    dy = ty[1] - ty[0] if len(ty) > 1 else 0.0
    dz = tz[1] - tz[0]
    xc, yc, zd = si.x_centers_mm, si.y_centers_mm, si.z_depths_mm
    x_all = matlab_colon(xc.min() + tx[0], dx, xc.max() + tx[-1] + dx / 2)
    y_all = matlab_colon(yc.min() + ty[0], dy, yc.max() + ty[-1] + dy / 2)
    z_all = matlab_colon(zd.min() + tz[0], dz, zd.max() + tz[-1] + dz / 2)
    zero = int(np.argmin(np.abs(z_all)))
    z_all = dz * (np.arange(1, len(z_all) + 1) - (zero + 1))
    if len(xc) == 1:
        x_all = tx.copy()
    if len(yc) == 1:
        y_all = ty.copy()
    z_full = z_all.copy()

    if output_pixel_size_um is not None:
        px = np.round(np.mean(np.diff(x_all)) * 1e3 * 100) / 100
        py = np.round(np.mean(np.diff(y_all)) * 1e3 * 100) / 100 if len(y_all) >= 2 else px
        if not (px == py == output_pixel_size_um):
            raise ValueError(f"outputFilePixelSize_um={output_pixel_size_um} must equal scan pixel "
                             f"size ({px}, {py}) um - legacy restriction")
        z_all = matlab_colon(z_all[0], output_pixel_size_um * 1e-3, z_all.max())

    if crop_z_range_mm is not None and len(crop_z_range_mm):
        lo, hi = crop_z_range_mm
        z_all = z_all[~((z_all < lo) | (z_all > hi))]
        if z_all.size == 0:
            raise ValueError("cropZRange_mm does not overlap the available z range")

    return TiledGeometry(tx, ty, tz, x_all, y_all, z_all, z_full, focus)
