"""Output writers compatible with myOCT's yOCT2Tif / yOCTFromTif.

Legacy format (yOCT2Tif.m + yOCT2Tif_ConvertBitsData.m):
  * BigTIFF, one page per output y plane, page = (nZ rows, nX cols) uint16
  * PackBits compression
  * bits = uint16(round((dB - c1) / (c2 - c1) * 65534)) + 1, NaN -> 0
  * TIFF tag 305 "Software" = JSON {"metadata": dim-struct, "clim": [c1 c2], "version": 3}
    (yOCTFromTif reads it from the first page only)
  * legacy clim = [min, max] of all finite dB values in the volume

Legacy also quantises twice (per-plane clim, then re-quantised to the global
clim in partialFileMode 3). `legacy_double_quantization=True` reproduces that
for bit-level comparisons; default is a single (more accurate) quantisation.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import tifffile

MAXBIT = 2 ** 16 - 1


def _matlab_uint16(v):
    """MATLAB uint16(): round half away from zero, saturate, NaN -> 0."""
    out = np.floor(np.abs(v) + 0.5) * np.sign(v)
    out = np.nan_to_num(out, nan=0.0)
    return np.clip(out, 0, MAXBIT).astype(np.uint16)


def db_to_bits(db: np.ndarray, clim) -> np.ndarray:
    c1, c2 = sorted(clim)
    bits = _matlab_uint16((db - c1) / (c2 - c1) * (MAXBIT - 1)).astype(np.uint32) + 1
    bits = np.minimum(bits, MAXBIT).astype(np.uint16)
    bits[np.isnan(db)] = 0
    return bits


def bits_to_db(bits: np.ndarray, clim) -> np.ndarray:
    c1, c2 = sorted(clim)
    out = (bits.astype(np.float64) - 1) * (c2 - c1) / (MAXBIT - 1) + c1
    out[bits == 0] = np.nan
    return out


def plane_clim(db: np.ndarray):
    fin = db[np.isfinite(db)]
    return (float(fin.min()), float(fin.max())) if fin.size else (np.nan, np.nan)


def write_legacy_tiff(path: str | Path, db_volume, clim, metadata: dict,
                      legacy_double_quantization: bool = False, plane_clims=None, progress=None):
    """db_volume: array-like (nY, nZ, nX) float32 (may be a np.memmap)."""
    path = Path(path)
    meta = {"metadata": metadata, "clim": [float(clim[0]), float(clim[1])], "version": 3}
    meta_json = json.dumps(meta, separators=(",", ":"))
    with tifffile.TiffWriter(path, bigtiff=True) as tw:
        for yi in range(db_volume.shape[0]):
            plane = np.asarray(db_volume[yi], dtype=np.float64)
            if legacy_double_quantization:
                pc = plane_clims[yi] if plane_clims is not None else plane_clim(plane)
                plane = bits_to_db(db_to_bits(plane, pc), pc)
            bits = db_to_bits(plane, clim)
            tw.write(bits, compression="packbits", photometric="minisblack",
                     software=meta_json if yi == 0 else None, metadata=None, contiguous=False)
            if progress:
                progress(yi)
    with open(path.with_suffix(path.suffix + ".json"), "w") as f:
        json.dump(meta, f)
