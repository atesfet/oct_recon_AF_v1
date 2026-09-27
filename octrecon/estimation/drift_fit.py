"""Faithful port of yOCTMeasureFocusDrift_fitDrift.m (myOCT, repository root).

Fits a physically constrained, robust line to focus measurements ("clicks") taken at
several stage depths and returns a focus pixel for every z-depth of the scan.

  * Theil-Sen fit (median of all pairwise slopes, median intercept)   l.343-367
  * iterative residual rejection: threshold max(3*1.4826*MAD, 15 px), <= 6 rounds  l.97-139
  * slope standard error (Theil-Sen efficiency 0.91), R^2              l.141-168
  * clamp to physical slope range [0, (ns_max^2-ni^2)/(ni*na)/dz]      l.98-100, 170-172
  * regimes: normal | no-measurable-drift | clamped-to-max | suspicious-structure-clicks  l.174-219
  * tissue RI  n_s = sqrt(ni^2 + (m*dz)*ni*na), floored at ni          l.221-238
  * hinge: no drift for z <= 0, focus = round(m*max(z,0) + b), clamped to [1, nZ]  l.255-268
  * per-depth standard error with pi/2 (median) and 0.91 corrections   l.270-288

Units: stage z in mm (input) / um (internally), pixels 1-based like MATLAB.
"""
from __future__ import annotations

import numpy as np

REJECT_FLOOR_PIX = 15
MAX_REJECT_ITERATIONS = 6


def _matlab_round(x):
    x = np.asarray(x, float)
    return np.sign(x) * np.floor(np.abs(x) + 0.5)


def theil_sen_fit(z_um, pix):
    """-> (slope, intercept, has_slope). All unordered pairs at distinct z vote."""
    z_um = np.asarray(z_um, float)
    pix = np.asarray(pix, float)
    n = len(z_um)
    ii, jj = np.triu_indices(n, k=1)
    dz = z_um[jj] - z_um[ii]
    valid = np.abs(dz) > 1e-9
    if not np.any(valid):
        return np.nan, np.nan, False
    slopes = (pix[jj[valid]] - pix[ii[valid]]) / dz[valid]
    slope = float(np.median(slopes))
    intercept = float(np.median(pix - slope * z_um))
    return slope, intercept, True


def fit_drift(z_clicked_stage_mm, focus_clicked_pix, z_all_depths_mm, z_pixel_size_um, n_z_pixels,
              tissue_refractive_index=1.4, immersion_refractive_index=1.33,
              max_tissue_refractive_index=1.55, v=False):
    """Port of yOCTMeasureFocusDrift_fitDrift. Returns (focus_pix_per_depth (int ndarray), diagnostics dict)."""
    na = float(tissue_refractive_index)
    ni = float(immersion_refractive_index)
    ns_max = float(max_tissue_refractive_index)
    dz_um = float(z_pixel_size_um)
    nZ = int(_matlab_round(n_z_pixels))
    if dz_um <= 0 or nZ < 1:
        raise ValueError("z_pixel_size_um must be > 0 and n_z_pixels >= 1")
    msgs = []

    # ---- clean input clicks
    z_clicked_um = np.atleast_1d(np.asarray(z_clicked_stage_mm, float)).ravel() * 1e3
    pix_clicked = np.atleast_1d(np.asarray(focus_clicked_pix, float)).ravel()
    bad = np.isnan(z_clicked_um) | np.isnan(pix_clicked)
    if bad.any():
        msgs.append(f"{int(bad.sum())} click(s) had NaN values and were ignored.")
    in_gel = (z_clicked_um < 0) & ~bad
    if in_gel.any():
        msgs.append(f"{int(in_gel.sum())} click(s) at stage z < 0 were ignored: above the tissue "
                    "the focus does not drift, so they do not inform the tissue slope.")
    use_i = np.nonzero(~bad & ~in_gel)[0]
    z = z_clicked_um[use_i]
    pix = pix_clicked[use_i]
    n = len(z)
    if n == 0:
        raise ValueError("No usable focus measurements were provided. "
                         "The focus was not visible/clickable on any tile.")

    # ---- robust fit with iterative physical outlier rejection
    m_min = 0.0
    m_max = (ns_max ** 2 - ni ** 2) / (ni * na) / dz_um
    kept = np.arange(n)
    rejected = np.zeros(0, int)
    rejected_res = np.zeros(0)
    if n == 1:
        slope_meas, intercept = 0.0, float(pix[0])
        msgs.append("Only one focus click: assuming no drift (constant focus).")
    else:
        for it in range(1, MAX_REJECT_ITERATIONS + 2):
            slope_meas, intercept, has_slope = theil_sen_fit(z[kept], pix[kept])
            if not has_slope:
                slope_meas = 0.0
                intercept = float(np.median(pix[kept]))
                msgs.append("All clicks are at the same stage z: assuming no drift (constant focus).")
                break
            if len(kept) < 3 or it > MAX_REJECT_ITERATIONS:
                break
            r = pix[kept] - (slope_meas * z[kept] + intercept)
            sigma_mad = 1.4826 * np.median(np.abs(r - np.median(r)))
            thr = max(3 * sigma_mad, REJECT_FLOOR_PIX)
            is_out = np.abs(r) > thr
            if not is_out.any():
                break
            rejected = np.concatenate([rejected, kept[is_out]])
            rejected_res = np.concatenate([rejected_res, r[is_out]])
            kept = kept[~is_out]

    # ---- slope uncertainty and R^2
    n_kept = len(kept)
    r2 = np.nan
    se_slope = np.nan
    if n_kept >= 3:
        r_meas = pix[kept] - (slope_meas * z[kept] + intercept)
        sigma_res_meas = np.sqrt(np.sum(r_meas ** 2) / (n_kept - 2))
        sxx = np.sum((z[kept] - z[kept].mean()) ** 2)
        se_slope = sigma_res_meas / np.sqrt(0.91 * sxx) if sxx > 0 else np.nan
        ss_tot = np.sum((pix[kept] - pix[kept].mean()) ** 2)
        if ss_tot > 0:
            r2 = 1 - np.sum(r_meas ** 2) / ss_tot

    # ---- clamp to physically possible range + regime
    slope_used = min(max(slope_meas, m_min), m_max)
    was_clamped = slope_used != slope_meas
    structure_floor = 0.1 * (ni / na) / dz_um
    if np.isnan(se_slope) or se_slope <= 0:
        slope_sigmas = np.nan
        is_structure = slope_meas < -0.5 * (ni / na) / dz_um
    else:
        slope_sigmas = slope_meas / se_slope
        is_structure = slope_sigmas < -3 and slope_meas < -structure_floor

    if is_structure:
        regime = "suspicious-structure-clicks"
    elif abs(slope_sigmas) < 2:          # NaN compares False, like MATLAB
        regime = "no-measurable-drift"
    elif was_clamped and slope_meas > 0:
        regime = "clamped-to-max"
    else:
        regime = "normal"

    if was_clamped:
        intercept = float(np.median(pix[kept] - slope_used * z[kept]))
    if regime == "suspicious-structure-clicks":
        msgs.append(f"Measured drift slope ({slope_meas * dz_um:.3f} um/um) is significantly negative, which is "
                    "not physical (tissue index cannot be below the immersion medium). Clamped to 0, but "
                    "REVIEW THE MEASUREMENT: negative apparent drift means clicks landed on a fixed structure "
                    "(e.g. coverslip) instead of the focus.")
    elif regime == "no-measurable-drift":
        if slope_meas < 0:
            msgs.append(f"No measurable focus drift: slope {slope_meas * dz_um:.3f} um/um is zero within "
                        f"noise ({abs(slope_sigmas):.1f} sigma). Using a constant focus, which is the correct "
                        "answer for an index-matched or weakly mismatched sample.")
    elif regime == "clamped-to-max":
        msgs.append(f"Measured drift slope ({slope_meas * dz_um:.3f} um/um) exceeds the physical maximum for "
                    f"tissue index {ns_max:.2f}. Clamped to {m_max * dz_um:.3f} um/um.")

    ns_sq = ni ** 2 + (slope_meas * dz_um) * ni * na
    tissue_ri = np.nan if regime == "suspicious-structure-clicks" else float(np.sqrt(max(ns_sq, ni ** 2)))
    if not np.isnan(tissue_ri) and not np.isnan(se_slope) and tissue_ri > 0:
        se_tissue_ri = (se_slope * dz_um) * ni * na / (2 * tissue_ri)
    else:
        se_tissue_ri = np.nan

    reasons = []
    for k in range(len(rejected)):
        zr = z[rejected[k]] * 1e-3
        if rejected_res[k] < 0:
            reasons.append(f"click at z={zr:.3f} mm is {abs(rejected_res[k]):.0f} pixels above the drift line: "
                           "consistent with a click on a fixed structure (coverslip/tissue feature), "
                           "which moves opposite to the focus.")
        else:
            reasons.append(f"click at z={zr:.3f} mm is {rejected_res[k]:.0f} pixels below the drift line (outlier).")
        msgs.append("Rejected " + reasons[-1])

    # ---- evaluate the line at every requested depth (hinge at z = 0)
    z_all_um = np.atleast_1d(np.asarray(z_all_depths_mm, float)).ravel() * 1e3
    z_eff = np.maximum(z_all_um, 0)
    focus = _matlab_round(slope_used * z_eff + intercept)
    oor = (focus < 1) | (focus > nZ)
    if oor.any():
        msgs.append(f"{int(oor.sum())} depth(s) had their focus clamped to the image boundary. "
                    "The focus is not usable there; consider not scanning that deep.")
    focus = np.clip(focus, 1, nZ).astype(int)

    if n_kept >= 3:
        r_kept = pix[kept] - (slope_used * z[kept] + intercept)
        sigma_res = np.sqrt(np.sum(r_kept ** 2) / (n_kept - 2))
        zbar = z[kept].mean()
        sxx = np.sum((z[kept] - zbar) ** 2)
        if sxx > 0:
            se_at_depth = sigma_res * np.sqrt((np.pi / 2) / n_kept + (z_eff - zbar) ** 2 / (0.91 * sxx))
        else:
            se_at_depth = sigma_res * np.sqrt(np.pi / 2) * np.ones_like(z_eff) / np.sqrt(n_kept)
    else:
        se_at_depth = np.full(z_eff.shape, np.nan)

    max_measured_z_mm = float(np.max(z[kept]) * 1e-3)
    max_req = float(np.max(z_all_um) * 1e-3)
    if max_req - max_measured_z_mm > 0.1:
        deepest = int(np.argmax(z_all_um))
        msgs.append(f"Focus measured down to z={max_measured_z_mm:.2f} mm but the scan reaches z={max_req:.2f} mm: "
                    f"below {max_measured_z_mm:.2f} mm the focus position is an extrapolation "
                    f"(estimated error at the deepest tile: +/-{2 * se_at_depth[deepest]:.0f} pixels). "
                    "Measure as deep as the focus is visible to reduce this.")

    diag = {
        "slopeMeasured_pixPerUm": float(slope_meas),
        "slopeUsed_pixPerUm": float(slope_used),
        "driftSlope": float(slope_used * dz_um),
        "intercept_pix": float(intercept),
        "tissueRI": tissue_ri,
        "keptIdx": use_i[kept].tolist(),          # 0-based indices into the input clicks
        "rejectedIdx": use_i[rejected].tolist(),
        "rejectedReason": reasons,
        "wasSlopeClamped": bool(was_clamped),
        "driftRegime": regime,
        "seSlope_pixPerUm": float(se_slope),
        "seTissueRI": float(se_tissue_ri),
        "r2": float(r2),
        "seAtDepth_pix": se_at_depth.tolist(),
        "maxMeasuredZ_mm": max_measured_z_mm,
        "zPixelSize_um": dz_um,
        "messages": msgs,
    }
    if v:
        s = f"Focus drift fit: slope = {diag['driftSlope']:.3f}"
        if not np.isnan(se_slope):
            s += f" +/- {se_slope * dz_um:.3f}"
        s += f" um/um ({slope_used * 1e3:.1f} pix/mm)"
        if not np.isnan(tissue_ri):
            s += f", tissue RI = {tissue_ri:.3f}"
            if not np.isnan(se_tissue_ri):
                s += f" +/- {se_tissue_ri:.3f}"
        print(s + f". Kept {n_kept} of {n} clicks.")
        for m in msgs:
            print("  - " + m)
    return focus, diag
