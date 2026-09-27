"""Automatic focus-position detection for tiled z-stacked scans.

mode='legacy'  Non-interactive port of Processing/yOCTFindFocusTilledScan.m
               (manualRefinment=false):
  step 1 (l.57-59, 78-127): volume at the zDepth closest to 0 (focus at the gel/tissue
         interface); 5 y-frames round(linspace(1, nY, 5)); meanAbs (yOCTProcessScan,
         NO optical-path correction - it is commented out there, l.270-274); per frame:
         median over x, top round(nZ/3) pixels ignored, first argmax; focus1 = round(median).
  step 2 (l.61-71, 134-149): volume one step above (gel only): profile
         mean_y(median_x(meanAbs)), imgaussfilt(sigma=4) (17-tap, replicate padding),
         findpeaks, the peak closest to focus1.
  The same pixel is returned for every zDepth (no drift).
  Legacy quirk: the tile folder is scanInfo.octFolders{zDepthIndex} (l.59/71), i.e. the
  first xy tile when z is the innermost scan loop.  tile_selection='legacy' reproduces
  that; the default 'center' uses the xy tile closest to the scan origin (same depth).

mode='measure' (default, 'auto'; alias 'drift') Automatic version of yOCTMeasureFocusDrift.m,
               the tool that produced zChosenFocusPositions.mat: depths from
               firstMeasureDepth_um=50 every focusMeasureStep_um=50 (selectDepthsToMeasure,
               shallow scans fall back to z >= 0), central tile, optical-path corrected
               B-scans around the central B-scan (reconstructCenterBScan).  The user's click
               is replaced by: at z ~ 0 the gel/tissue interface (brightest x-median pixel,
               step-1 rule); deeper, the confocal peak of the detrended depth profile nearest
               the running prediction.  Clicks -> yOCTMeasureFocusDrift_fitDrift (drift_fit.py):
               constant focus for a single click / no measurable drift, per-depth otherwise.

Diagnostic (confocal_check=True): stationary_confocal_peak - the only depth feature that
does not move with the stage; pointwise median of the depth profiles of all zDepths.
"""
from __future__ import annotations

import time

import numpy as np

from ..params import resolve_dispersion
from ._common import VolumeContext, matlab_round, to_py
from .drift_fit import fit_drift

DRIFT_MIN_DEPTH_UM = 50.0


# --------------------------------------------------------------------------- MATLAB helpers
def imgaussfilt_1d(v, sigma=4.0):
    """imgaussfilt(v, sigma) for a vector: kernel size 2*ceil(2*sigma)+1, replicate padding."""
    v = np.asarray(v, float).ravel()
    h = int(np.ceil(2 * sigma))
    x = np.arange(-h, h + 1)
    k = np.exp(-x ** 2 / (2 * sigma ** 2))
    k /= k.sum()
    p = np.concatenate([np.full(h, v[0]), v, np.full(h, v[-1])])
    return np.convolve(p, k, mode="valid")


def findpeaks_matlab(v):
    """Indices (0-based) of MATLAB findpeaks local maxima: strictly larger than the
    neighbours, flat peaks reported at their first sample, endpoints excluded."""
    v = np.asarray(v, float)
    n = len(v)
    peaks = []
    i = 1
    while i < n - 1:
        if v[i] > v[i - 1]:
            j = i
            while j < n - 1 and v[j + 1] == v[i]:
                j += 1
            if j < n - 1 and v[j + 1] < v[i]:
                peaks.append(i)
            i = j + 1
        else:
            i += 1
    return np.asarray(peaks, int)


def _nanargmax_first(v):
    return int(np.nanargmax(v))   # numpy returns the first occurrence, like find(..,1,'first')


def _to_host(a):
    return a.get() if hasattr(a, "get") else np.asarray(a)


# --------------------------------------------------------------------------- building blocks
def legacy_frames(n_y: int):
    """yToLoad = unique(round(linspace(1, nY, 5))) -> 0-based frames."""
    return (np.unique(matlab_round(np.linspace(1, n_y, 5))).astype(int) - 1).tolist()


def _tile_for_depth(ctx, zi: int, tile_selection: str):
    si = ctx.si
    if tile_selection == "legacy":
        folders = si.json["octFolders"]
        folders = [folders] if isinstance(folders, str) else folders
        return folders[min(zi, len(folders) - 1)]
    xi0, yi0 = ctx.central_indices()
    t = ctx.by_index.get((xi0, yi0, zi))
    if t is None:
        t = next(t for t in si.tiles if t.zi == zi)
    return t.folder


def mean_abs_bscans(ctx, folder, frames, dispersion, opc=False):
    """meanAbs of the given frames -> host array (nFrames, nX, nZ) float64 (NaN where OPC invalid)."""
    mag = _to_host(ctx.magnitude(folder, frames, dispersion)).astype(float)
    if opc:
        mag = apply_opc(ctx, mag, frames)
    return mag


def apply_opc(ctx, mag, frames):
    """yOCTOpticalPathCorrection (nearest-neighbour, NaN outside) on (F, nX, nZ) tile B-scans."""
    from ..core.geometry import build_tiled_geometry
    from ..core.stitching import OpticalPathCorrection
    poly = ctx.si.optical_path_polynomial()
    if poly is None:
        return mag
    sg = ctx.geometry(0.0)
    tg = build_tiled_geometry(ctx.si, sg.z_um, np.full(len(ctx.si.z_depths_mm), np.nan), None, None)
    oc = OpticalPathCorrection.build(poly, tg.tile_x_mm, tg.tile_y_mm, tg.tile_z_mm)
    nz = mag.shape[-1]
    rr = np.arange(nz)
    out = np.empty_like(mag)
    for i, f in enumerate(frames):
        src = rr[None, :] + oc.shift[f][:, None]
        t = rr[None, :] + oc.pdz[f][:, None]
        valid = (t >= 0) & (t <= nz - 1)
        g = np.take_along_axis(mag[i], np.clip(src, 0, nz - 1), axis=-1)
        out[i] = np.where(valid, g, np.nan)
    return out


def legacy_focus(ctx, dispersion, tile_selection="center", log=print):
    """Steps 1 and 2 of yOCTFindFocusTilledScan. Returns dict."""
    si = ctx.si
    zd = si.z_depths_mm
    idx1 = int(np.argmin(np.abs(zd - 0)))
    idx2 = min(max(1, idx1 - 1), len(zd) - 1)          # MATLAB min(max(2, i1-1), n) in 0-based
    frames = legacy_frames(ctx.n_frames)
    nz = ctx.n_z

    folder1 = _tile_for_depth(ctx, idx1, tile_selection)
    m1 = mean_abs_bscans(ctx, folder1, frames, dispersion)        # (F, nX, nZ)
    tissue_zi = []
    med_profiles1 = []
    for i in range(len(frames)):
        med = np.median(m1[i], axis=0)                             # median over x
        med_profiles1.append(med.copy())
        med[: int(matlab_round(nz / 3))] = np.nan
        tissue_zi.append(_nanargmax_first(med) + 1)                # 1-based
    focus1 = int(matlab_round(np.median(tissue_zi)))
    log(f"focus step 1: tile {folder1} (z={zd[idx1] * 1e3:.0f} um) brightest pixel per frame "
        f"{tissue_zi} -> {focus1}")

    folder2 = _tile_for_depth(ctx, idx2, tile_selection)
    m2 = mean_abs_bscans(ctx, folder2, frames, dispersion)
    prof = np.median(m2, axis=1).mean(axis=0)                      # median x, mean y
    smooth = imgaussfilt_1d(prof, 4)
    p = findpeaks_matlab(smooth) + 1                               # 1-based
    if len(p):
        focus2 = int(matlab_round(p[int(np.argmin(np.abs(p - focus1)))]))
    else:
        focus2 = focus1
    log(f"focus step 2: tile {folder2} (z={zd[idx2] * 1e3:.0f} um, gel) nearest smoothed peak -> {focus2}")
    return {
        "step1": {"tile": folder1, "z_depth_mm": float(zd[idx1]), "frames": frames,
                  "per_frame_pix": tissue_zi, "value": focus1,
                  "profile": np.mean(med_profiles1, axis=0).tolist()},
        "step2": {"tile": folder2, "z_depth_mm": float(zd[idx2]), "frames": frames,
                  "profile": prof.tolist(), "profile_smoothed": smooth.tolist(),
                  "peaks": p.tolist(), "value": focus2},
        "focus": focus2,
    }


def select_depths_to_measure(z_depths_mm, step_um=50.0, first_um=50.0):
    """Port of selectDepthsToMeasure (yOCTMeasureFocusDrift.m l.126-162); 0-based indices."""
    z = np.asarray(z_depths_mm, float)
    d = np.median(np.abs(np.diff(z))) if len(z) > 1 else 0
    stride = 1 if d <= 0 else max(1, int(matlab_round(step_um * 1e-3 / d)))
    cand = np.nonzero(z >= first_um * 1e-3 - 1e-9)[0]
    fallback = False
    if cand.size == 0:
        fallback = True
        cand = np.nonzero(z >= -1e-9)[0]
        if cand.size == 0:
            raise ValueError("All zDepths of this scan are above the tissue (z < 0); no focus drift to measure.")
    cand = cand[np.argsort(z[cand], kind="stable")]
    return cand[::stride].tolist(), fallback


def detect_peak_near(profile, guess_pix, search_px=60, baseline_sigma=30.0, smooth_sigma=4.0):
    """Confocal peak in a depth profile: log, remove slow attenuation trend, smooth, peak
    nearest to guess (1-based) within +-search_px.  Returns (pix or nan, prominence_dB)."""
    prof = np.asarray(profile, float)
    lp = 10 * np.log10(np.where(np.isfinite(prof) & (prof > 0), prof, np.nan))
    ok = np.isfinite(lp)
    if ok.sum() < 10:
        return np.nan, 0.0
    lp = np.interp(np.arange(len(lp)), np.nonzero(ok)[0], lp[ok])
    detr = imgaussfilt_1d(lp, smooth_sigma) - imgaussfilt_1d(lp, baseline_sigma)
    p = findpeaks_matlab(detr) + 1
    p = p[np.abs(p - guess_pix) <= search_px]
    if p.size == 0:
        return np.nan, 0.0
    best = p[np.argmax(detr[p - 1])]
    return int(best), float(detr[best - 1])


def interface_pix(mag_opc):
    """Step-1 rule on (optical-path corrected) B-scans: per frame brightest pixel of the
    x-median profile below the top third; median over frames (1-based)."""
    nz = mag_opc.shape[-1]
    vals = []
    for m in mag_opc:
        med = np.nanmedian(m, axis=0)
        med[: int(matlab_round(nz / 3))] = np.nan
        vals.append(_nanargmax_first(med) + 1)
    return int(matlab_round(np.median(vals))), vals


def measure_focus_per_depth(ctx, dispersion, prior_pix, log=print, step_um=50.0, first_um=50.0,
                            n_frames=5, search_px=60, min_prominence_db=0.3):
    """Automatic yOCTMeasureFocusDrift: measure the focus on the central tile of the depths
    chosen by selectDepthsToMeasure, then fit with yOCTMeasureFocusDrift_fitDrift.

    Per measured depth ("click" replacement), on optical-path corrected B-scans around the
    central B-scan (like reconstructCenterBScan):
      * zDepth ~ 0 (only reached by the legacy shallow-scan fallback): the protocol puts the
        gel/tissue interface at the focus -> brightest interface pixel (step-1 rule).
      * deeper: confocal peak of the detrended depth profile closest to the running prediction.
    """
    si = ctx.si
    zd = si.z_depths_mm
    idxs, fallback = select_depths_to_measure(zd, step_um, first_um)
    if fallback:
        log(f"measure: no zDepths >= {first_um:.0f} um; measuring from z = 0 (legacy shallow-scan fallback)")
    step_mm = float(np.median(np.abs(np.diff(zd)))) if len(zd) > 1 else 0.0
    xi0, yi0 = ctx.central_indices()
    c = (ctx.n_frames - 1) / 2
    half = max(1, int(round(0.05 * ctx.n_frames)))
    frames = np.unique(np.clip(np.round(np.linspace(c - half, c + half, n_frames)), 0, ctx.n_frames - 1)
                       .astype(int)).tolist()
    sg = ctx.geometry(dispersion)
    dz_um = float(np.median(np.diff(sg.z_um)))
    clicks_z, clicks_pix, per = [], [], []
    guess = prior_pix
    for zi in idxs:
        z = float(zd[zi])
        t = ctx.by_index.get((xi0, yi0, zi)) or next(t for t in si.tiles if t.zi == zi)
        m = mean_abs_bscans(ctx, t.folder, frames, dispersion, opc=True)
        with np.errstate(all="ignore"):
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                prof = np.nanmean(np.nanmedian(m, axis=1), axis=0)
                if abs(z) <= 0.5 * step_mm + 1e-9:
                    pix, frame_vals = interface_pix(m)
                    rule, prom, acc = "interface (brightest, z~0)", None, True
                else:
                    pix, prom = detect_peak_near(prof, guess, search_px)
                    frame_vals = None
                    rule = "confocal peak (detrended profile)"
                    acc = bool(np.isfinite(pix) and prom >= min_prominence_db)
        per.append({"zi": int(zi), "z_depth_mm": z, "tile": t.folder, "frames": frames, "rule": rule,
                    "detected_pix": None if not np.isfinite(pix) else int(pix), "per_frame_pix": frame_vals,
                    "prominence_dB": prom, "accepted": acc, "profile": prof.tolist()})
        log(f"measure: z={z * 1e3:.0f} um tile {t.folder} [{rule}]: {pix}{'' if acc else ' - skipped'}")
        if acc:
            clicks_z.append(z)
            clicks_pix.append(float(pix))
            if len(clicks_z) >= 2:
                guess = float(fit_drift(clicks_z, clicks_pix, [z + step_um * 1e-3], dz_um, ctx.n_z,
                                        tissue_refractive_index=si.tissue_ri)[0][0])
            else:
                guess = pix
    if not clicks_z:
        raise RuntimeError("focus measurement: the focus was not detected on any measured depth")
    focus, diag = fit_drift(clicks_z, clicks_pix, zd, dz_um, ctx.n_z, tissue_refractive_index=si.tissue_ri)
    return focus, diag, per


def stationary_confocal_peak(ctx, dispersion, tiles_xy=None, frames=None, log=print):
    """Diagnostic: the focus is the only depth feature that does NOT move with the stage.
    Pointwise median (over all zDepths) of the dB depth profiles removes the moving
    reflectors/tissue surface; the detrended maximum is the confocal peak (1-based)."""
    import warnings
    si = ctx.si
    xi0, yi0 = ctx.central_indices()
    tiles_xy = tiles_xy or [(xi0, yi0)]
    frames = frames or [int(round(f * (ctx.n_frames - 1))) for f in (0.25, 0.5, 0.75)]
    P = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for zi in range(len(si.z_depths_mm)):
            acc = []
            for xy in tiles_xy:
                t = ctx.by_index.get((xy[0], xy[1], zi))
                if t is None:
                    continue
                m = mean_abs_bscans(ctx, t.folder, frames, dispersion, opc=True)
                acc.append(np.nanmean(np.nanmedian(m, axis=1), axis=0))
            if acc:
                P.append(20 * np.log10(np.nanmean(acc, axis=0)))
        P = np.array(P)
        med = np.nanmedian(P, axis=0)
    med = np.interp(np.arange(len(med)), np.nonzero(np.isfinite(med))[0], med[np.isfinite(med)])
    d = imgaussfilt_1d(med, 4) - imgaussfilt_1d(med, 50)
    lo = int(matlab_round(len(d) / 3))
    pk = findpeaks_matlab(d) + 1
    pk = pk[pk > lo]
    best = int(pk[np.argmax(d[pk - 1])]) if pk.size else int(np.argmax(d[lo:]) + lo + 1)
    log(f"stationary confocal peak (median over {len(P)} depths): {best}")
    return {"value": best, "prominence_dB": float(d[best - 1]), "profile_detrended": d.tolist(),
            "tiles_xy": [list(map(int, xy)) for xy in tiles_xy], "frames": frames}


# --------------------------------------------------------------------------- main API
def detect_focus(volume_folder, dispersion_quadratic_term=None, device="cpu", mode="auto",
                 tile_selection="center", confocal_check=True, log=print) -> dict:
    """Detect the focus pixel (1-based, one per zDepth) of a tiled scan.

    dispersion_quadratic_term : None/'auto' -> params.resolve_dispersion (the legacy tools
        take it via reconstructConfig; it sharpens the peaks).
    mode : 'auto' (= 'measure') | 'measure' | 'drift' (alias of 'measure') | 'legacy'
        (see module docstring).  The legacy steps 1/2 and (confocal_check) the stationary
        confocal peak are always computed and returned as diagnostics.
    tile_selection : legacy steps only: 'center' (default) | 'legacy' (octFolders{zDepthIndex}).
    """
    log = log or (lambda m: None)
    t0 = time.perf_counter()
    ctx = VolumeContext(volume_folder, device=device)
    try:
        si = ctx.si
        if dispersion_quadratic_term in (None, "", "auto"):
            disp, disp_src = resolve_dispersion(si)
        else:
            disp, disp_src = float(dispersion_quadratic_term), "user"
        zd = si.z_depths_mm
        pos_range_um = float(max(zd.max(), 0) * 1e3)
        if mode in ("auto", "measure", "drift"):
            mode_used = "measure"
        elif mode == "legacy":
            mode_used = "legacy"
        else:
            raise ValueError("mode must be auto|measure|drift|legacy")
        log(f"focus detection: mode={mode_used} (positive depth range {pos_range_um:.0f} um, drift "
            f"{'measured' if pos_range_um >= DRIFT_MIN_DEPTH_UM - 1e-6 else 'not measurable: shallow scan'}) "
            f"dispersion={disp:.5g} ({disp_src})")

        leg = legacy_focus(ctx, disp, tile_selection, log)
        sg = ctx.geometry(disp)
        dz_um = float(np.median(np.diff(sg.z_um)))
        out = {"step1": leg["step1"]["value"], "step2": leg["step2"]["value"],
               "legacy": {k: leg[k] for k in ("step1", "step2")},
               "dispersion_quadratic_term": disp, "dispersion_source": disp_src,
               "z_pixel_size_um": dz_um, "n_z": ctx.n_z, "z_depths_mm": zd.tolist(),
               "positive_depth_range_um": pos_range_um,
               "tile_selection": tile_selection, "device": ctx.device}
        if mode_used == "legacy":
            focus = [leg["focus"]] * len(zd)
            out["method"] = ("yOCTFindFocusTilledScan port (step 1 interface brightest pixel, "
                             "step 2 nearest smoothed gel peak; manual step skipped) - constant")
            out["per_depth"] = [{"zi": i, "z_depth_mm": float(z), "focus_pix": leg["focus"],
                                 "source": "constant (legacy)"} for i, z in enumerate(zd)]
        else:
            focus, diag, per = measure_focus_per_depth(ctx, disp, leg["step1"]["value"], log)
            focus = [int(f) for f in focus]
            deep = pos_range_um >= DRIFT_MIN_DEPTH_UM - 1e-6
            out["method"] = ("automatic yOCTMeasureFocusDrift: "
                             + ("confocal peak on central tiles every 50 um from 50 um" if deep else
                                "shallow scan -> interface at z=0 on the central B-scans (legacy fallback)")
                             + " + yOCTMeasureFocusDrift_fitDrift")
            out["drift"] = {"measurements": per, "fit": diag}
            out["per_depth"] = [{"zi": i, "z_depth_mm": float(z), "focus_pix": focus[i],
                                 "se_pix": diag["seAtDepth_pix"][i], "source": "drift fit"}
                                for i, z in enumerate(zd)]
        if confocal_check and len(zd) >= 3:
            try:
                out["confocal_peak"] = stationary_confocal_peak(ctx, disp, log=log)
            except Exception as e:   # diagnostic only
                out["confocal_peak"] = {"error": str(e)}
        out["focus_positions"] = [int(f) for f in focus]
        out["mode"] = mode_used
        out["elapsed_s"] = time.perf_counter() - t0
        log(f"focus positions: {out['focus_positions']} ({out['elapsed_s']:.1f} s)")
        return to_py(out)
    finally:
        ctx.close()
