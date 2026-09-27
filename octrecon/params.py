"""Volume inspection and automatic parameter resolution.

`inspect_volume()` reads only small metadata files (ScanInfo.json, one Header.xml,
focus .mat) and returns everything the UI / CLI needs: scan summary, storage layout,
and an automatically resolved value + provenance for every reconstruction parameter.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from . import presets
from .io.scaninfo import ScanInfo
from .io.volume import TileReader, scan_volume_layout

FOCUS_FILE = "zChosenFocusPositions.mat"


def load_focus_file(path: str | Path) -> np.ndarray:
    import scipy.io as sio
    return np.asarray(sio.loadmat(str(path))["focusPositionInImageZpix"], float).ravel()


def resolve_dispersion(si: ScanInfo, value="auto"):
    """-> (value, source)."""
    if value not in (None, "", "auto", "estimate"):
        return float(value), "user"
    pre = presets.probe_preset(si.oct_system, si.probe)
    if pre and "dispersion_quadratic_term" in pre:
        return float(pre["dispersion_quadratic_term"]), \
            f"probe preset ({si.oct_system}, {si.probe.get('ObjectiveName')})"
    if "DefaultDispersionQuadraticTerm" in (si.probe or {}):
        return float(si.probe["DefaultDispersionQuadraticTerm"]), "ScanInfo.json octProbe.DefaultDispersionQuadraticTerm"
    return presets.LEGACY_DEFAULT_DISPERSION, "legacy default (yOCTProcessTiledScan)"


def resolve_focus_sigma(si: ScanInfo, value="auto"):
    if value not in (None, "", "auto"):
        return float(value), "user"
    pre = presets.probe_preset(si.oct_system, si.probe)
    if pre and "focus_sigma" in pre:
        return float(pre["focus_sigma"]), "probe preset"
    mag = presets.magnification_from_probe(si.probe, si.json.get("octProbePath", ""))
    if mag in presets.FOCUS_SIGMA_BY_MAGNIFICATION:
        return presets.FOCUS_SIGMA_BY_MAGNIFICATION[mag], f"objective {mag} rule (legacy demo comments)"
    return presets.LEGACY_DEFAULT_FOCUS_SIGMA, "legacy default (yOCTProcessTiledScan)"


def resolve_crop(si: ScanInfo, value="auto"):
    if value in ("auto", None, ""):
        return [float(si.z_depths_mm.min()), float(si.z_depths_mm.max())], "[min, max] of ScanInfo zDepths (legacy demo rule)"
    if value == "none":
        return None, "user: no crop"
    return [float(value[0]), float(value[1])], "user"


def resolve_pixel_size(si: ScanInfo, value="auto"):
    if value in ("auto", None, ""):
        return float(si.pixel_size_um), "ScanInfo.json pixelSize_um"
    return float(value), "user"


def resolve_n(si: ScanInfo, value="auto"):
    if value in ("auto", None, ""):
        return float(si.tissue_ri), "ScanInfo.json tissueRefractiveIndex"
    return float(value), "user"


def resolve_focus(si: ScanInfo, volume: Path, value="auto"):
    """-> (focus array per depth or None if it must be estimated, source).
    'auto': focus file in the volume folder if present, else automatic detection.
    'file:<path>' / path ending in .mat, number, list, 'estimate', 'none'."""
    nd = len(si.z_depths_mm)
    if value in ("auto", None, ""):
        p = volume / FOCUS_FILE
        if p.exists():
            return load_focus_file(p), f"{FOCUS_FILE} in volume folder"
        return None, "automatic detection (no focus file found)"
    if value == "estimate":
        return None, "automatic detection"
    if value == "none":
        return np.full(nd, np.nan), "user: no focus gating"
    if isinstance(value, str) and (value.startswith("file:") or value.lower().endswith(".mat")):
        p = Path(value[5:] if value.startswith("file:") else value)
        p = p if p.is_absolute() else volume / p
        return load_focus_file(p), f"file {p.name}"
    arr = np.atleast_1d(np.asarray(value, float)).ravel()
    if arr.size == 1:
        arr = np.full(nd, arr[0])
    return arr, "user"


def inspect_volume(volume: str | Path) -> dict:
    volume = Path(volume)
    if not (volume / "ScanInfo.json").exists():
        # allow pointing at the parent folder that contains OCTVolume/
        if (volume / "OCTVolume" / "ScanInfo.json").exists():
            volume = volume / "OCTVolume"
        else:
            raise FileNotFoundError(f"No ScanInfo.json in {volume} (select the OCTVolume folder)")
    si = ScanInfo(volume)
    folders = [t.folder for t in si.tiles]
    layout = scan_volume_layout(volume, folders)
    tr = TileReader(volume / folders[0])
    h = tr.header
    tr.close()

    disp, disp_src = resolve_dispersion(si)
    sig, sig_src = resolve_focus_sigma(si)
    crop, crop_src = resolve_crop(si)
    px, px_src = resolve_pixel_size(si)
    n, n_src = resolve_n(si)
    focus, focus_src = resolve_focus(si, volume)

    n_bscans = len(folders) * h.size_y * h.bscan_avg
    raw_gb = n_bscans * h.bytes_per_file / 1e9
    nx_out = int(round((np.ptp(si.x_centers_mm) + si.tile_range_x_mm) / (px * 1e-3)))
    ny_out = int(round((np.ptp(si.y_centers_mm) + si.tile_range_y_mm) / (px * 1e-3)))
    nz_out = int(np.floor((crop[1] - crop[0]) / (px * 1e-3))) + 1 if crop else None
    return {
        "volume_folder": str(volume),
        "scan": {
            "oct_system": si.oct_system, "probe": si.probe.get("ObjectiveName"),
            "probe_path": si.json.get("octProbePath"),
            "n_tiles": len(folders), "grid_xyz": [len(si.x_centers_mm), len(si.y_centers_mm), len(si.z_depths_mm)],
            "tile_px": [si.n_x_px, si.n_y_px], "tile_mm": [si.tile_range_x_mm, si.tile_range_y_mm],
            "z_depths_mm": si.z_depths_mm.tolist(), "pixel_size_um": si.pixel_size_um,
            "spectral_px": h.n_lambda, "alines_per_bscan": h.interf_size, "apod_lines": h.apod_size,
            "ascan_avg": h.ascan_avg, "bscan_avg": h.bscan_avg, "spectra_avg": h.spectra_avg,
            "raw_size_GB": round(raw_gb, 1),
            "estimated_output_shape_yzx": [ny_out, nz_out, nx_out],
        },
        "storage": layout,
        "auto": {
            "dispersion_quadratic_term": {"value": disp, "source": disp_src},
            "focus_sigma": {"value": sig, "source": sig_src},
            "crop_z_range_mm": {"value": crop, "source": crop_src},
            "output_pixel_size_um": {"value": px, "source": px_src},
            "n_medium": {"value": n, "source": n_src},
            "focus_positions": {"value": None if focus is None else focus.tolist(), "source": focus_src},
            "interp_method": {"value": "sinc5", "source": "legacy demo default"},
        },
    }
