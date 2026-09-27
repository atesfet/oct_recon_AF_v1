"""Tests for octrecon.estimation (dispersion, focus, drift fit, previews).

Unit tests run everywhere. Real-data tests run only when OCT_TEST_VOLUME points to an
OCTVolume folder (validated on 10um_FOV_1):
    OCT_TEST_VOLUME=/path/10um_FOV_1/OCTVolume pytest -q tests/test_estimation.py
Device: OCT_DEVICE=cpu|gpu (default cpu).
"""
import os
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from octrecon.estimation.drift_fit import fit_drift, theil_sen_fit  # noqa: E402
from octrecon.estimation.focus import findpeaks_matlab, imgaussfilt_1d, select_depths_to_measure  # noqa: E402

VOL = os.environ.get("OCT_TEST_VOLUME")
DEVICE = os.environ.get("OCT_DEVICE", "cpu")
needs_volume = pytest.mark.skipif(not VOL or not (Path(VOL) / "ScanInfo.json").exists(),
                                  reason="set OCT_TEST_VOLUME to an OCTVolume folder")
REF_DISPERSION = 8.949e7     # reproduces the legacy 10um_FOV_1 reconstruction bit-exactly
REF_FOCUS = 433              # zChosenFocusPositions.mat (one click at z=0)
DISP_REL_TOL = 0.01          # measured: -0.51 % (8.903e7) from the probe .ini default start
FOCUS_TOL_PX = 3             # measured: +2 px (435)


# --------------------------------------------------------------------------- drift fit
DZ = 1.4339
NZ = 1024


def test_theil_sen_exact_line():
    z = np.array([50, 100, 150, 200.0])
    s, b, ok = theil_sen_fit(z, 0.2 * z + 400)
    assert ok and np.isclose(s, 0.2) and np.isclose(b, 400)
    assert theil_sen_fit([10, 10], [1, 2])[2] is False


def test_drift_synthetic_normal_with_outlier():
    rng = np.random.default_rng(0)
    na, ni, ns = 1.4, 1.33, 1.45
    m = (ns ** 2 - ni ** 2) / (ni * na) / DZ               # true slope pix/um
    z_mm = np.arange(0.05, 0.401, 0.05)
    pix = 430 + m * z_mm * 1e3 + rng.normal(0, 1.0, z_mm.size)
    pix[3] -= 60                                            # click on a fixed structure
    zall = np.arange(-0.1, 0.4001, 0.01)
    focus, d = fit_drift(z_mm, pix, zall, DZ, NZ, tissue_refractive_index=na)
    assert d["driftRegime"] == "normal"
    assert d["rejectedIdx"] == [3]
    assert abs(d["slopeUsed_pixPerUm"] - m) < 0.05 * m
    assert abs(d["tissueRI"] - ns) < 0.02
    assert np.all(focus[zall <= 0] == focus[0])             # hinge: flat above tissue
    exp = np.round(430 + m * np.maximum(zall, 0) * 1e3)
    assert np.max(np.abs(focus - exp)) <= 2
    assert np.all(np.isfinite(d["seAtDepth_pix"]))


def test_drift_no_drift_and_gel_clicks_ignored():
    z_mm = np.array([-0.05, 0.05, 0.1, 0.15, 0.2])
    pix = np.array([300, 433.4, 432.6, 433.2, 432.9])
    focus, d = fit_drift(z_mm, pix, [0, 0.1, 0.2], DZ, NZ, tissue_refractive_index=1.33)
    assert d["driftRegime"] == "no-measurable-drift"
    assert 0 not in d["keptIdx"] and any("stage z < 0" in m for m in d["messages"])
    assert np.all(focus == 433)


def test_drift_single_click_and_clamps():
    focus, d = fit_drift([0.0], [433], np.arange(-0.03, 0.041, 0.01), DZ, NZ)
    assert np.all(focus == 433) and d["driftSlope"] == 0
    # slope far above the physical maximum -> clamped-to-max
    z = np.array([0.05, 0.1, 0.15, 0.2])
    pix = 400 + 2.0 * z * 1e3 + np.array([0.3, -0.2, 0.1, -0.1])
    _, d = fit_drift(z, pix, [0.1], DZ, NZ)
    assert d["driftRegime"] == "clamped-to-max" and d["wasSlopeClamped"]
    # strongly negative slope -> structure clicks, RI NaN
    pix = 500 - 0.8 * z * 1e3 + np.array([0.3, -0.2, 0.1, -0.1])
    _, d = fit_drift(z, pix, [0.1], DZ, NZ)
    assert d["driftRegime"] == "suspicious-structure-clicks" and np.isnan(d["tissueRI"])
    with pytest.raises(ValueError):
        fit_drift([np.nan], [1], [0], DZ, NZ)


def test_matlab_helpers():
    v = np.array([0, 1, 3, 3, 2, 5, 1, 1, 4.0])
    assert findpeaks_matlab(v).tolist() == [2, 5]           # flat peak -> first sample; no endpoints
    s = imgaussfilt_1d(np.r_[np.zeros(20), 1, np.zeros(20)], 4)
    assert np.isclose(s.sum(), 1) and np.argmax(s) == 20
    assert np.allclose(imgaussfilt_1d(np.ones(10), 4), 1)   # replicate padding
    idx, fb = select_depths_to_measure(np.arange(-0.03, 0.0401, 0.01))
    assert fb and idx == [3]
    idx, fb = select_depths_to_measure(np.arange(0, 0.5001, 0.01))
    assert not fb and idx == [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]


def test_dispersion_window_matches_reconstruction():
    """Scorer window == build_spectral_geometry window (only beta changes)."""
    from octrecon.core.geometry import build_spectral_geometry
    lam = np.linspace(800, 1000, 256)
    sg0 = build_spectral_geometry(lam, 0.0, 1.33, "sinc5")
    sg = build_spectral_geometry(lam, 8.949e7, 1.33, "sinc5")
    k = 2 * np.pi / sg0.lambda_eq_nm
    w = sg0.window * np.exp(-1j * 8.949e7 * (k - k.mean()) ** 2)
    assert np.allclose(w, sg.window, rtol=1e-9, atol=1e-9)


def test_render_png_synthetic():
    from octrecon.estimation import preview
    img = np.random.default_rng(0).normal(0, 1, (128, 64)).astype(np.float32)
    png = preview.render_png(img, focus_pix=60)
    assert png[:8] == b"\x89PNG\r\n\x1a\n"


# --------------------------------------------------------------------------- real data
@needs_volume
def test_estimate_dispersion_real():
    from octrecon.estimation.dispersion import estimate_dispersion
    from octrecon.estimation import preview
    r = estimate_dispersion(VOL, device=DEVICE, initial=8.6883e7, log=None)   # probe .ini default start
    rel = r["value"] / REF_DISPERSION - 1
    print(f"dispersion {r['value']:.5g} rel err {rel * 100:.2f}% in {r['elapsed_s']:.1f}s")
    assert abs(rel) < DISP_REL_TOL
    assert not r["at_bound"]
    assert r["elapsed_s"] < 60
    assert len(r["per_bscan_values"]) == len(r["tiles_used"]) == 8
    assert preview.dispersion_curve_png(r)[:4] == b"\x89PNG"


@needs_volume
def test_detect_focus_real():
    from octrecon.estimation.focus import detect_focus
    from octrecon.estimation import preview
    r = detect_focus(VOL, dispersion_quadratic_term=REF_DISPERSION, device=DEVICE, log=None)
    fp = np.array(r["focus_positions"])
    print("focus", fp, "legacy steps", r["step1"], r["step2"], "confocal", r["confocal_peak"]["value"])
    assert len(set(fp.tolist())) == 1                       # no drift on this shallow scan
    assert abs(fp[0] - REF_FOCUS) <= FOCUS_TOL_PX
    assert preview.focus_profile_png(r)[:4] == b"\x89PNG"


@needs_volume
def test_preview_real():
    from octrecon.estimation import preview
    p = preview.preview_bscan(VOL, 3, dispersion_quadratic_term=REF_DISPERSION, device=DEVICE)
    assert p["image_db"].shape[1] <= 500 and p["image_db"].shape[0] == p["z_px_count"]
    assert preview.render_png(p["image_db"], p["focus_hint"])[:4] == b"\x89PNG"
