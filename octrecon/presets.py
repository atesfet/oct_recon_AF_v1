"""System / probe constants that are not stored in the raw data.

These values are constant for a given OCT system + probe (objective + scan head)
and are therefore offered as *advanced options* with automatic defaults.

Resolution order for the dispersion term (dispersionQuadraticTerm, nm^2/rad):
  1. PROBE_PRESETS[(system, ObjectiveName)]  -- validated values (this table)
  2. ScanInfo.json octProbe.DefaultDispersionQuadraticTerm   -- probe .ini default
  3. LEGACY_DEFAULT_DISPERSION (yOCTProcessTiledScan.m default)
Add a line to PROBE_PRESETS once a value has been validated for a probe
(e.g. with the automatic dispersion search in octrecon.estimation).
"""
from __future__ import annotations

LEGACY_DEFAULT_DISPERSION = 79430000.0   # yOCTProcessTiledScan.m:55
LEGACY_DEFAULT_FOCUS_SIGMA = 20.0        # yOCTProcessTiledScan.m:56

# (octSystem lower-case, octProbe.ObjectiveName) -> constants
PROBE_PRESETS = {
    ("gan632", "Olympus20xOCTGWinter"): {
        "dispersion_quadratic_term": 8.949e7,
        "focus_sigma": 10.0,
        "note": "validated: reproduces the legacy MATLAB reconstruction of 10um_FOV_1 bit-exactly "
                "(Demo_ScanAndProcess3D.m lists 8.962e7; probe .ini default is 8.6883e7)",
    },
}

# focusSigma per objective magnification (comments in Demo_ScanAndProcess3D.m /
# Demo_CheckFocusInTileScan.m: 10x -> 20, 20x -> 10, 40x -> 10 (or 5))
FOCUS_SIGMA_BY_MAGNIFICATION = {"10x": 20.0, "20x": 10.0, "40x": 10.0}


def probe_preset(oct_system: str, probe: dict) -> dict | None:
    name = (probe or {}).get("ObjectiveName", "")
    return PROBE_PRESETS.get(((oct_system or "").lower(), name))


def magnification_from_probe(probe: dict, probe_path: str = "") -> str | None:
    txt = f"{(probe or {}).get('ObjectiveName', '')} {probe_path}".lower()
    for m in ("10x", "20x", "40x"):
        if m in txt:
            return m
    return None
