"""Wasatch raw spectra (octSystem 'Wasatch').

3D mode (legacy yOCTLoadInterfFromFile_WasatchHeader.m l.38-61, _WasatchData.m):
    %05d_raw_us_<nLambda>_<nX>_<nBScanAvg>.bin     one file per B-scan (repeat), index
        0-based = (y-1)*nBScanAvg + (b-1); uint16, lambda-major per A-line (MATLAB
        reshape(in, nLambda, nX)); no apodization lines.
    nY = nFiles / nBScanAvg. Background ("apodization") = mean over all A-lines of the
    frames loaded together (the raw_bg_00001.tif branch always fails in 3D mode because it
    is read with the .bin reader) -> apod_mode 'frame_mean'.
2D mode (l.15-36): raw_%05d.tif (1-based, image rows = A-lines, columns = lambda),
    nBScanAvg = number of raw_0*.tif files, nY = 1, background raw_bg_00001.tif (mean over
    its rows) when present, else frame mean.
Wavelength: fixed polynomial lambda(nm) = cl0 + cl1*p + cl2*p^2 + cl3*p^3, p = 1..nLambda.
"""
from __future__ import annotations

import re
import threading
from pathlib import Path

import numpy as np

from .thorlabs import TileHeader
from .volume import TileReader

BIN_RE = re.compile(r"^(\d{5})_raw_us_(\d+)_(\d+)_(\d+)\.bin$")
TIF_RE = re.compile(r"^raw_0\d*\.tif$")
WASATCH_LAMBDA_POLY = (6.57328e02, 8.28908e-02, -1.10175e-06, -7.04714e-11)


def wasatch_lambda_nm(n_lambda: int) -> np.ndarray:
    cl0, cl1, cl2, cl3 = WASATCH_LAMBDA_POLY
    p = np.arange(1, n_lambda + 1, dtype=np.float64)
    return cl0 + cl1 * p + cl2 * p ** 2 + cl3 * p ** 3


class WasatchTileReader(TileReader):
    def __init__(self, tile_dir: str | Path):
        self.dir = Path(tile_dir)
        self._zip = None
        self._members = None
        self._lock = threading.Lock()
        tifs = sorted(p.name for p in self.dir.glob("raw_0*.tif") if TIF_RE.match(p.name))
        self.bg = None
        if tifs:
            import tifffile
            self.layout = "wasatch_tif"
            first = tifffile.imread(self.dir / tifs[0])
            size_x, n = first.shape                       # Height = A-lines, Width = lambda
            n_files, size_y = len(tifs), 1
            b_avg = n_files
            bgp = self.dir / "raw_bg_00001.tif"
            if bgp.exists():
                self.bg = np.asarray(tifffile.imread(bgp))
            apod = 0 if self.bg is None else self.bg.shape[0]
            raw_dtype = "<u2" if first.dtype in (np.uint8, np.uint16) else "<f4"
            self.header = TileHeader(
                n_lambda=n, interf_size=apod + size_x, apod_size=apod, size_x=size_x,
                size_y=size_y, size_x_mm=float("nan"), size_y_mm=float("nan"), ascan_avg=1,
                bscan_avg=b_avg, spectra_avg=1, model="Wasatch", raw_dtype=raw_dtype,
                apod_mode="lines" if self.bg is not None else "frame_mean", manufacturer="Wasatch")
            return
        bins = sorted(p.name for p in self.dir.glob("*_raw_us_*.bin"))
        if not bins:
            raise FileNotFoundError(f"{self.dir}: no Wasatch raw files")
        m = BIN_RE.match(bins[0])
        if not m:
            raise ValueError(f"{bins[0]}: expected %05d_raw_us_<nLambda>_<nX>_<nBScanAvg>.bin")
        self.layout = "wasatch_bin"
        n, size_x, b_avg = int(m.group(2)), int(m.group(3)), int(m.group(4))
        if len(bins) % b_avg:
            raise ValueError(f"{self.dir}: {len(bins)} files is not a multiple of nBScanAvg={b_avg}")
        self.header = TileHeader(
            n_lambda=n, interf_size=size_x, apod_size=0, size_x=size_x, size_y=len(bins) // b_avg,
            size_x_mm=float("nan"), size_y_mm=float("nan"), ascan_avg=1, bscan_avg=b_avg,
            spectra_avg=1, model="Wasatch", raw_dtype="<u2", apod_mode="frame_mean",
            manufacturer="Wasatch")

    def chirp(self, fallback=None):
        raise NotImplementedError("Wasatch data has no chirp; use lambda_nm()")

    def lambda_nm(self, oct_system: str = "Wasatch", data_dir=None) -> np.ndarray:
        return wasatch_lambda_nm(self.header.n_lambda)

    def file_name(self, file_index: int) -> str:
        h = self.header
        if self.layout == "wasatch_tif":
            return "raw_%05d.tif" % (int(file_index) + 1)   # 2D: file index starts at 1
        return "%05d_raw_us_%d_%d_%d.bin" % (int(file_index), h.n_lambda, h.size_x, h.bscan_avg)

    def read_bscans(self, file_indices, out: np.ndarray) -> np.ndarray:
        h = self.header
        a = h.apod_size if h.apod_mode == "lines" else 0
        for k, fi in enumerate(file_indices):
            p = self.dir / self.file_name(fi)
            if self.layout == "wasatch_tif":
                import tifffile
                img = np.asarray(tifffile.imread(p))
                if img.shape != (h.size_x, h.n_lambda):
                    raise IOError(f"{p}: expected {(h.size_x, h.n_lambda)} image, got {img.shape}")
                if a:
                    out[k, :a] = self.bg
                out[k, a:] = img
            else:
                data = np.fromfile(p, dtype="<u2")
                if data.size != h.n_lambda * h.size_x:
                    raise IOError(f"{p}: expected {h.n_lambda * h.size_x} samples, got {data.size}")
                out[k] = data.reshape(h.size_x, h.n_lambda)
        return out
