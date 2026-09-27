"""Thorlabs SRR raw files (octSystem 'Ganymede_SRR' / 'Telesto_SRR').

One file per B-scan (and B-scan repeat), named
    Data_Y%04d_YTotal%d_B%04d_BTotal%d_<System>.srr      (System = Ganymede | Telesto)
File content:
    line 1          'headersize=<n>\\n'
    n bytes         whitespace separated 'key=value' tokens, e.g. size1=2048 (spectral
                    pixels), scanregions=<k>,<start>,<end>, aporegions=<k>,<start>,<end>
                    (0-based start, exclusive end, in A-lines)
    rest            uint16 samples, A-line major (size1 samples per A-line); only the low
                    12 bits are data (legacy: rem(data, 4096)); exactly size1*scanend values.
A text chirp file (Chirp.dat / chirp.dat, one value per line) may sit in the folder;
otherwise the chirp shipped with myOCT is used.

Legacy: LoadSave/yOCTLoadInterfFromFile_ThorlabsSRRHeader.m, _ThorlabsSRRData.m.
Returned raw arrays are int16 (values 0..4095 after rem) with the apodization A-lines
first, i.e. the same (interf_size, n_lambda) layout as a Thorlabs Spectral*.data file.
"""
from __future__ import annotations

import re
import threading
from pathlib import Path

import numpy as np

from .thorlabs import TileHeader
from .volume import TileReader

SRR_NAME_RE = re.compile(r"^Data_Y(\d{1,4})_YTotal(\d+)_B(\d{1,4})_BTotal(\d+)_(.+)\.srr$")


def parse_srr_header(b: bytes) -> dict:
    """readSRRHeader (yOCTLoadInterfFromFile_ThorlabsSRRHeader.m l.108-144)."""
    nl = b.index(b"\n")
    first = b[:nl].decode("ascii", errors="replace").strip()
    m = re.match(r"headersize=(\d+)", first)
    if not m:
        raise ValueError("SRR file does not start with 'headersize=<n>'")
    hsize = int(m.group(1))
    total = nl + 1 + hsize
    txt = b[nl + 1: total].decode("ascii", errors="replace")
    h = {"headerTotalBytes": total}
    for tok in txt.split():
        if "=" not in tok:
            continue
        k, v = tok.split("=", 1)
        if k == "scanregions":
            p = v.split(",")
            h["scanstart"], h["scanend"] = int(float(p[1])), int(float(p[2]))
        elif k == "aporegions":
            p = v.split(",")
            h["apodstart"], h["apodend"] = int(float(p[1])), int(float(p[2]))
        else:
            try:
                h[k] = float(v) if any(c in v for c in ".eE") else int(v)
            except ValueError:
                h[k] = v.strip("'\"")
    return h


def srr_file_name(y1: int, y_total: int, b1: int, b_total: int, system: str) -> str:
    """ThorlabsSRRData.m l.65-70 (y1, b1 are 1-based; '_SRR' removed from the system)."""
    return "Data_Y%04d_YTotal%d_B%04d_BTotal%d_%s.srr" % (y1, y_total, b1, b_total,
                                                          system.replace("_SRR", "").replace("_srr", ""))


class SRRTileReader(TileReader):
    def __init__(self, tile_dir: str | Path):
        self.dir = Path(tile_dir)
        self.layout = "srr"
        self._zip = None
        self._members = None
        self._lock = threading.Lock()
        files = sorted(p.name for p in self.dir.glob("*.srr"))
        if not files:
            raise FileNotFoundError(f"{self.dir}: no .srr files")
        # legacy reads the header of the first file of the datastore (sorted order)
        m = SRR_NAME_RE.match(files[0])
        if not m:
            raise ValueError("SRR file formating is wrong. This code expects this file name format: "
                             "Data_Y%04d_YTotal%d_B%04d_BTotal%d_%s.srr")
        self.y_total, self.b_total, self.file_system = int(m.group(2)), int(m.group(4)), m.group(5)
        with open(self.dir / files[0], "rb") as f:
            head = f.read(1 << 16)
        self.srr = h = parse_srr_header(head)
        n = int(h["size1"])
        apod = h["apodend"] - h["apodstart"]
        size_x = h["scanend"] - h["scanstart"]
        if apod <= 0:
            raise ValueError(f"{self.dir}: SRR file has no apodization region")
        self.header = TileHeader(
            n_lambda=n, interf_size=apod + size_x, apod_size=apod, size_x=size_x,
            size_y=self.y_total, size_x_mm=float("nan"), size_y_mm=float("nan"),
            ascan_avg=1, bscan_avg=self.b_total, spectra_avg=1, model=self.file_system,
            raw_dtype="<i2", apod_mode="lines", manufacturer="Thorlabs_SRR")
        self.file_lines = int(h["scanend"])   # legacy expects exactly size1*scanend samples

    @property
    def oct_system(self) -> str:
        return self.file_system + "_SRR"

    def file_name(self, file_index: int) -> str:
        y0, b0 = divmod(int(file_index), self.b_total)
        return srr_file_name(y0 + 1, self.y_total, b0 + 1, self.b_total, self.file_system)

    def read_file(self, file_index: int) -> np.ndarray:
        """-> (file_lines, n_lambda) int16 raw samples (after rem 4096)."""
        h, n = self.srr, self.header.n_lambda
        p = self.dir / self.file_name(file_index)
        b = p.read_bytes()
        data = np.frombuffer(b, dtype="<u2", offset=h["headerTotalBytes"],
                             count=(len(b) - h["headerTotalBytes"]) // 2)
        if data.size != n * self.file_lines:
            raise IOError(f"{p}: expected {n * self.file_lines} samples, got {data.size}")
        return (data % 4096).astype(np.int16).reshape(self.file_lines, n)

    def read_bscans(self, file_indices, out: np.ndarray) -> np.ndarray:
        h = self.srr
        a = self.header.apod_size
        for k, fi in enumerate(file_indices):
            lines = self.read_file(fi)
            out[k, :a] = lines[h["apodstart"]:h["apodend"]]
            out[k, a:] = lines[h["scanstart"]:h["scanend"]]
        return out
