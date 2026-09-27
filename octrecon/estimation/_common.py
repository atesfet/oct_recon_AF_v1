"""Shared helpers for the estimators: volume access and per-B-scan processing.

Processing here is identical to the reconstruction path (octrecon.core.spectral):
apodization subtraction, sinc5 k-linearisation, Hann/rms * exp(-i*beta*(k-k0)^2),
ifft, |.|, mean over B-scan repeats.  Legacy equivalents: yOCTLoadInterfFromFile.m +
yOCTInterfToScanCpx.m (+ meanAbs in yOCTProcessScan.m).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ..backend import get_xp
from ..core.geometry import build_spectral_geometry
from ..core.spectral import SpectralProcessor, average_repeats
from ..io.scaninfo import ScanInfo, Tile
from ..io.volume import TileReader

DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def resolve_volume(volume_folder) -> Path:
    v = Path(volume_folder)
    if not (v / "ScanInfo.json").exists() and (v / "OCTVolume" / "ScanInfo.json").exists():
        v = v / "OCTVolume"
    if not (v / "ScanInfo.json").exists():
        raise FileNotFoundError(f"No ScanInfo.json in {v} (select the OCTVolume folder)")
    return v


def matlab_round(x):
    """MATLAB round(): halves away from zero."""
    x = np.asarray(x, float)
    return np.sign(x) * np.floor(np.abs(x) + 0.5)


class VolumeContext:
    """Lazily opened tile readers + spectral geometry for one tiled volume."""

    def __init__(self, volume_folder, device: str = "cpu", dtype=np.float32):
        self.volume = resolve_volume(volume_folder)
        self.si = ScanInfo(self.volume)
        self.xp, self.device = get_xp(device)
        self.dtype = np.dtype(dtype)
        self._readers: dict[str, TileReader] = {}
        first = self.reader(self.si.tiles[0].folder)
        self.hdr = first.header
        self.lambda_nm = first.lambda_nm(self.si.oct_system, DATA_DIR)
        self._sg_cache = {}
        self.by_index = {(t.xi, t.yi, t.zi): t for t in self.si.tiles}

    # ------------------------------------------------------------------ access
    def reader(self, folder: str) -> TileReader:
        r = self._readers.get(folder)
        if r is None:
            r = self._readers[folder] = TileReader(self.volume / folder)
        return r

    def close(self):
        for r in self._readers.values():
            r.close()
        self._readers.clear()

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()

    @property
    def n_frames(self) -> int:
        return int(self.hdr.size_y)

    @property
    def n_z(self) -> int:
        return len(self.lambda_nm) // 2

    def tile(self, folder: str) -> Tile:
        for t in self.si.tiles:
            if t.folder == folder:
                return t
        raise KeyError(folder)

    def central_indices(self):
        """(xi0, yi0): tile closest to the scan origin (legacy yOCTMeasureFocusDrift)."""
        return int(np.argmin(np.abs(self.si.x_centers_mm))), int(np.argmin(np.abs(self.si.y_centers_mm)))

    def read_raw(self, folder: str, frames) -> np.ndarray:
        """Raw int16 (nFrames*bscan_avg, interfSize, nLambda); repeats are consecutive files."""
        n = max(1, int(self.hdr.bscan_avg))
        files = [int(f) * n + a for f in frames for a in range(n)]
        buf = np.empty((len(files), self.hdr.interf_size, self.hdr.n_lambda), np.dtype(self.hdr.raw_dtype))
        self.reader(folder).read_bscans(files, buf)
        return buf

    # --------------------------------------------------------------- processing
    def geometry(self, dispersion: float, n_medium: float | None = None):
        n_medium = self.si.tissue_ri if n_medium is None else float(n_medium)
        key = (float(dispersion), n_medium)
        sg = self._sg_cache.get(key)
        if sg is None:   # ~30 ms; cached per (dispersion, n)
            sg = self._sg_cache[key] = build_spectral_geometry(self.lambda_nm, dispersion, n_medium, "sinc5")
        return sg

    def processor(self, dispersion: float, n_medium: float | None = None) -> SpectralProcessor:
        h = self.hdr
        return SpectralProcessor(self.geometry(dispersion, n_medium), h.apod_size, xp=self.xp, dtype=self.dtype,
                                 ascan_binning=h.spectra_avg, apod_mode=h.apod_mode, apod_group=max(1, h.bscan_avg))

    def magnitude(self, folder: str, frames, dispersion: float, n_medium: float | None = None,
                  sp: SpectralProcessor | None = None):
        """|scan| averaged over B-scan repeats -> (nFrames, nX, nZ) on the backend (xp)."""
        sp = sp or self.processor(dispersion, n_medium)
        raw = self.xp.asarray(self.read_raw(folder, frames))
        mag = average_repeats(sp.magnitude(raw), len(frames), self.hdr)
        return mag


def to_py(v):
    """numpy scalars/arrays -> plain Python (JSON-friendly)."""
    if isinstance(v, dict):
        return {k: to_py(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [to_py(x) for x in v]
    if isinstance(v, np.ndarray):
        return to_py(v.tolist())
    if isinstance(v, np.generic):
        return v.item()
    if isinstance(v, float) and not np.isfinite(v):
        return None
    return v
