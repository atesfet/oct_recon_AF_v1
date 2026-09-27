"""Reader for Thorlabs (Ganymede / GAN632) raw spectral tile folders.

Folder layout of one tile (e.g. OCTVolume/Data01):
    Header.xml             OCITY XML: sizes, apodization region, instrument info
    data/Chirp.data        float32[2048]  spectrometer pixel -> k mapping
    data/OffsetErrors.data float32[2048]  (not used by myOCT reconstruction)
    data/Spectral{i}.data  int16[interfSize, 2048] (C order) = one raw B-scan,
                           i = 0..nY*nBScanAvg-1, first `apodSize` A-lines are
                           the apodization (galvo parked, no sample) lines.

Equivalent MATLAB: yOCTLoadInterfFromFile_ThorlabsHeader.m,
yOCTLoadInterfFromFile_ThorlabsLoadChirp.m, yOCTLoadInterfFromFile_ThorlabsData.m
"""
from __future__ import annotations

import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class TileHeader:
    n_lambda: int          # spectrometer pixels (2048)
    interf_size: int       # A-lines per raw B-scan file incl. apodization (525)
    apod_size: int         # apodization A-lines at the start of each file (25)
    size_x: int            # scan A-lines (500)
    size_y: int            # B-scans (500)
    size_x_mm: float
    size_y_mm: float
    ascan_avg: int
    bscan_avg: int
    spectra_avg: int       # "AScanBinning"
    model: str
    # --- generalisations for non-Thorlabs-OCITY backends (defaults = OCITY behaviour) ---
    raw_dtype: str = "<i2"         # numpy dtype of the arrays returned by read_bscans
    apod_mode: str = "lines"       # "lines": first apod_size A-lines of every file are the
                                   #   apodization; "frame_mean": no apodization lines, the
                                   #   background is the mean of all A-lines of the loaded y
                                   #   frame (Wasatch, yOCTLoadInterfFromFile_WasatchData.m l.115-131)
    manufacturer: str = "Thorlabs"  # Thorlabs | Thorlabs_SRR | Wasatch

    @property
    def bytes_per_file(self) -> int:
        return self.n_lambda * self.interf_size * np.dtype(self.raw_dtype).itemsize

    @property
    def n_scan_lines(self) -> int:
        """A-lines per file after the apodization lines (before AScanBinning)."""
        return self.interf_size - (self.apod_size if self.apod_mode == "lines" else 0)


def read_header(tile_folder: str | Path) -> TileHeader:
    return parse_header_xml((Path(tile_folder) / "Header.xml").read_bytes())


def parse_header_xml(xml_bytes: bytes) -> TileHeader:
    root = ET.fromstring(xml_bytes)
    spectral = None
    for df in root.find("DataFiles"):
        if (df.text or "").startswith("data\\Spectral") or (df.text or "").startswith("data/Spectral"):
            spectral = df
            break
    if spectral is None:
        raise ValueError("No Spectral data file entry in Header.xml")
    a = spectral.attrib
    img = root.find("Image")
    acq = root.find("Acquisition")
    ia = acq.find("IntensityAveraging")
    sa = acq.find("SpeckleAveraging")
    size_y = int(img.find("SizePixel/SizeY").text) if img.find("SizePixel/SizeY") is not None else 1
    sy_real = img.find("SizeReal/SizeY")
    size_y_mm = float(sy_real.text) if sy_real is not None else 0.0
    bscan_avg = int(sa.find("SlowAxis").text) if sa is not None else 1
    # yOCTLoadInterfFromFile_ThorlabsHeader.m l.97-131: no SizeReal/SizeY or SizeY==0 -> 2D
    # scan (one y frame). SizeY<0 -> "Y dimension is actually BScanAvg" (SlowAxis ignored).
    if size_y_mm < 0:
        bscan_avg = size_y
        size_y = 1
    elif size_y_mm == 0:
        size_y = 1
    return TileHeader(
        n_lambda=int(a["SizeZ"]),
        interf_size=int(a["SizeX"]),
        apod_size=int(a["ApoRegionEnd0"]),
        size_x=int(img.find("SizePixel/SizeX").text),
        size_y=size_y,
        size_x_mm=float(img.find("SizeReal/SizeX").text),
        size_y_mm=size_y_mm,
        ascan_avg=int(ia.find("AScans").text),
        bscan_avg=bscan_avg,
        spectra_avg=int(ia.find("Spectra").text),
        model=root.find("Instrument/Model").text,
    )


def detect_thorlabs_system(xml_bytes: bytes) -> str:
    """yOCTLoadInterfFromFile_WhatOCTSystemIsIt.m l.77-99: the WHOLE Header.xml text is
    searched for 'Ganymed' / 'Telesto'; exactly one must match. Note a GAN632 header
    (<Series>Ganymede</Series>) is therefore detected as 'Ganymede' (legacy behaviour)."""
    txt = xml_bytes.decode("utf-8", errors="replace")
    is_g, is_t = "Ganymed" in txt, "Telesto" in txt
    if is_g + is_t != 1:
        raise ValueError("Couldn't figure out what OCT system that is (Header.xml mentions "
                         f"Ganymed={is_g}, Telesto={is_t})")
    return "Ganymede" if is_g else "Telesto"


def read_chirp(tile_folder: str | Path, fallback: str | Path | None = None) -> np.ndarray:
    """Chirp.data is binary float32 (the legacy loader first tries a text parse,
    which fails on binary data, then falls back to float32)."""
    p = Path(tile_folder) / "data" / "Chirp.data"
    if p.exists():
        return np.fromfile(p, dtype="<f4").astype(np.float64)
    if fallback is not None:  # text version shipped with myOCT (ChirpGanymede.dat)
        return np.loadtxt(fallback, dtype=np.float64).ravel()
    raise FileNotFoundError(p)


def spectral_path(tile_folder: str | Path, file_index: int) -> str:
    return os.path.join(str(tile_folder), "data", f"Spectral{file_index}.data")


def read_bscans_into(tile_folder, file_indices, out: np.ndarray, hdr: TileHeader) -> np.ndarray:
    """Read raw B-scans into a preallocated int16 array of shape
    (len(file_indices), interf_size, n_lambda). Uses raw readinto (no copies).
    A missing / truncated file raises (legacy replaces it with NaN; we prefer
    to fail loudly and let the caller decide)."""
    nbytes = hdr.bytes_per_file
    for k, fi in enumerate(file_indices):
        path = spectral_path(tile_folder, fi)
        mv = memoryview(out[k]).cast("B")
        with open(path, "rb", buffering=0) as f:
            got = f.readinto(mv)
        if got != nbytes:
            raise IOError(f"{path}: expected {nbytes} bytes, got {got}")
    return out
