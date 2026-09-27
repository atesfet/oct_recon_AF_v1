"""Simulated tiles (octSystem 'Simulated Ganymede', legacy Simulation/yOCTSimulateTileScan.m).

Each tile folder holds data.mat with
    interf  single (nLambda, nX, nY[, nAScanAvg, nBScanAvg])  - interferogram, no apodization
    dim     dimensions struct (lambda.values in nm, x.index, y.index, [AScanAvg/BScanAvg])
Legacy: yOCTLoadInterfFromFile.m l.131-132 (header = dim), _SimulatedData.m (apodization =
zeros, i.e. nothing is subtracted). Here each "file" (y frame, B-scan repeat) is returned
as float32 (1 + nX*nAScanAvg, nLambda) with a zero apodization line first.
"""
from __future__ import annotations

import threading
from pathlib import Path

import numpy as np

from .thorlabs import TileHeader
from .volume import TileReader


def _load_mat(path: Path):
    try:
        import scipy.io as sio
        m = sio.loadmat(path, struct_as_record=False, squeeze_me=False)
        interf = np.asarray(m["interf"])
        dim = m["dim"][0, 0]

        def field(s, name):
            return getattr(s, name)[0, 0] if hasattr(s, name) else None
        lam = np.asarray(field(dim, "lambda").values, float).ravel()
        n_x = np.asarray(field(dim, "x").index).size
        n_y = np.asarray(field(dim, "y").index).size
        a = field(dim, "AScanAvg")
        b = field(dim, "BScanAvg")
        n_a = np.asarray(a.index).size if a is not None else 1
        n_b = np.asarray(b.index).size if b is not None else 1
    except NotImplementedError:                      # MATLAB v7.3 (HDF5) file
        import h5py
        with h5py.File(path, "r") as f:
            interf = np.asarray(f["interf"]).T
            d = f["dim"]
            lam = np.asarray(d["lambda"]["values"]).ravel().astype(float)
            n_x = np.asarray(d["x"]["index"]).size
            n_y = np.asarray(d["y"]["index"]).size
            n_a = np.asarray(d["AScanAvg"]["index"]).size if "AScanAvg" in d else 1
            n_b = np.asarray(d["BScanAvg"]["index"]).size if "BScanAvg" in d else 1
    interf = interf.reshape(interf.shape + (1,) * (5 - interf.ndim))
    if interf.shape != (len(lam), n_x, n_y, n_a, n_b):
        raise ValueError(f"{path}: interf shape {interf.shape} does not match dim "
                         f"{(len(lam), n_x, n_y, n_a, n_b)}")
    return interf, lam


class SimulatedTileReader(TileReader):
    def __init__(self, tile_dir: str | Path):
        self.dir = Path(tile_dir)
        self.layout = "simulated"
        self._zip = None
        self._members = None
        self._lock = threading.Lock()
        self._interf, self._lambda = _load_mat(self.dir / "data.mat")
        n, x, y, a, b = self._interf.shape
        self.header = TileHeader(
            n_lambda=n, interf_size=1 + x * a, apod_size=1, size_x=x, size_y=y,
            size_x_mm=float("nan"), size_y_mm=float("nan"), ascan_avg=a, bscan_avg=b, spectra_avg=1,
            model="Simulated Ganymede", raw_dtype="<f4", apod_mode="lines", manufacturer="Simulated")

    def chirp(self, fallback=None):
        raise NotImplementedError("simulated data has no chirp; use lambda_nm()")

    def lambda_nm(self, oct_system: str = "Simulated Ganymede", data_dir=None) -> np.ndarray:
        return self._lambda.copy()

    def read_bscans(self, file_indices, out: np.ndarray) -> np.ndarray:
        n, x, y, a, b = self._interf.shape
        for k, fi in enumerate(file_indices):
            y0, b0 = divmod(int(fi), b)
            frame = self._interf[:, :, y0, :, b0]                  # (N, X, A)
            out[k, 0] = 0
            out[k, 1:] = frame.transpose(1, 2, 0).reshape(x * a, n)  # A-line = x*A + a
        return out
