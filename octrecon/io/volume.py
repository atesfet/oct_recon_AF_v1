"""Tile access for tiled OCT volumes, independent of how the tiles are stored.

Supported tile layouts (the raw inputs the legacy myOCT loaders accept):
  * "unzipped": DataNN/Header.xml + DataNN/data/{Spectral*.data, Chirp.data, ...}
    (Thorlabs OCITY; Gan632 Python SDK output, or legacy yOCTUnzipTiledScan output)
  * "oct":      DataNN/VolumeGanymedeOCTFile.oct (or any single *.oct), a Thorlabs
    OCITY file = ZIP archive of the same Header.xml + data/* entries. Entry names may
    use Windows separators ("data\\Spectral0.data"). Read in place (no extraction,
    no extra disk space) or extracted like legacy yOCTUnzipTiledScan/yOCTUnzipOCTFolder.
  * "srr":      Thorlabs SRR files Data_Y%04d_YTotal%d_B%04d_BTotal%d_<System>.srr
    (+ optional text chirp file), see octrecon/io/srr.py
  * "wasatch":  Wasatch %05d_raw_us_<nLambda>_<nX>_<nBScanAvg>.bin (3D) or raw_%05d.tif
    (2D) files, see octrecon/io/wasatch.py
  * "simulated": data.mat (interf + dim) written by legacy yOCTSimulateTileScan
    ('Simulated Ganymede'), see octrecon/io/simulated.py

`TileReader(tile_dir)` picks the backend from the folder content and always exposes:
  .header  TileHeader (n_lambda, interf_size, apod_size, size_x, size_y, ascan_avg,
           bscan_avg, spectra_avg, raw_dtype, apod_mode, manufacturer, bytes_per_file)
  .chirp(fallback) / .lambda_nm(oct_system, data_dir)
  .read_bscans(file_indices, out)  -> out[k] = (interf_size, n_lambda) raw spectra of
           file k (file index = (y-1)*nBScanAvg + (bScanAvg-1), 0-based like Spectral{i})
  .close()

Legacy reference: LoadSave/yOCTUnzipTiledScan.m, LoadSave/yOCTUnzipOCTFolder.m,
LoadSave/yOCTLoadInterfFromFile*.m
"""
from __future__ import annotations

import io
import os
import re
import shutil
import threading
import zipfile
from pathlib import Path

import numpy as np

from .thorlabs import TileHeader, parse_header_xml

LEGACY_OCT_NAME = "VolumeGanymedeOCTFile.oct"
_SRR_GLOB = "Data_Y*.srr"
_WASATCH_BIN_RE = re.compile(r"^(\d{5})_raw_us_(\d+)_(\d+)_(\d+)\.bin$")
_WASATCH_TIF_RE = re.compile(r"^raw_0\d*\.tif$")

# Wavelength-range / packaged-chirp choice per system
# (yOCTLoadInterfFromFile_ThorlabsHeaderLambda.m l.9-24)
_CHIRP_FILE = {"ganymede": "ChirpGanymede.dat", "ganymede_srr": "ChirpGanymede.dat",
               "gan632": "ChirpGanymede.dat",
               "telesto": "ChirpTelesto.dat", "telesto_srr": "ChirpTelesto.dat"}


def _norm(name: str) -> str:
    return name.replace("\\", "/").lstrip("/")


def find_oct_archive(tile_dir: Path) -> Path | None:
    p = tile_dir / LEGACY_OCT_NAME
    if p.exists():
        return p
    cands = sorted(tile_dir.glob("*.oct"))
    return cands[0] if len(cands) == 1 else None


def tile_layout(tile_dir: Path) -> str:
    """'unzipped' | 'oct' | 'srr' | 'wasatch' | 'simulated' | 'missing'."""
    tile_dir = Path(tile_dir)
    if (tile_dir / "Header.xml").exists() and (tile_dir / "data").is_dir():
        return "unzipped"
    if not tile_dir.is_dir():
        return "missing"
    if find_oct_archive(tile_dir) is not None:
        return "oct"
    try:
        names = os.listdir(tile_dir)
    except OSError:
        return "missing"
    if any(n.startswith("Data_Y") and n.endswith(".srr") for n in names):
        return "srr"
    if any(_WASATCH_BIN_RE.match(n) or _WASATCH_TIF_RE.match(n) for n in names):
        return "wasatch"
    if "data.mat" in names:
        return "simulated"
    return "missing"


def parse_text_floats(b: bytes) -> np.ndarray:
    """MATLAB textscan(fid,'%f'): read whitespace separated numbers until the first
    token that is not a number."""
    vals = []
    for tok in b.split():
        try:
            vals.append(float(tok))
        except ValueError:
            break
    return np.asarray(vals, dtype=np.float64)


def decode_chirp_bytes(b: bytes) -> np.ndarray:
    """yOCTLoadInterfFromFile_ThorlabsLoadChirp.m l.52-62: first try text, if that gives
    fewer than 2 values read binary float32."""
    try:
        v = parse_text_floats(b)
    except Exception:
        v = np.zeros(0)
    if len(v) >= 2:
        return v
    return np.frombuffer(b[: len(b) // 4 * 4], dtype="<f4").astype(np.float64)


class TileReader:
    """Uniform access to one tile: header, chirp, raw B-scan files.

    Instantiating TileReader(tile_dir) returns the backend matching the folder content
    (this class itself handles Thorlabs OCITY tiles, unzipped or .oct)."""

    def __new__(cls, tile_dir=None, *args, **kwargs):
        if cls is TileReader and tile_dir is not None:
            lay = tile_layout(Path(tile_dir))
            if lay == "srr":
                from .srr import SRRTileReader
                cls = SRRTileReader
            elif lay == "wasatch":
                from .wasatch import WasatchTileReader
                cls = WasatchTileReader
            elif lay == "simulated":
                from .simulated import SimulatedTileReader
                cls = SimulatedTileReader
        return super().__new__(cls)

    def __init__(self, tile_dir: str | Path):
        self.dir = Path(tile_dir)
        self.layout = tile_layout(self.dir)
        if self.layout not in ("unzipped", "oct"):
            raise FileNotFoundError(f"{self.dir}: neither Header.xml+data/ nor a .oct archive found")
        self._zip = None
        self._lock = threading.Lock()
        self._members = None
        if self.layout == "oct":
            self._zip = zipfile.ZipFile(find_oct_archive(self.dir), "r")
            self._members = {_norm(i.filename): i for i in self._zip.infolist()}
        self.header: TileHeader = parse_header_xml(self.read_bytes("Header.xml"))

    # ---- raw access -----------------------------------------------------
    def exists(self, rel: str) -> bool:
        if self.layout == "oct":
            return _norm(rel) in self._members
        return (self.dir / rel).exists()

    def read_bytes(self, rel: str) -> bytes:
        if self.layout != "oct":
            return (self.dir / rel).read_bytes()
        with self._lock:  # ZipFile is not thread-safe for concurrent reads
            return self._zip.read(self._members[_norm(rel)])

    def list_files(self) -> list[str]:
        """All files of the tile (relative, '/' separated), recursively."""
        if self.layout == "oct":
            return [m for m in self._members if m and not m.endswith("/")]
        out = []
        for root, _, files in os.walk(self.dir):
            rel = os.path.relpath(root, self.dir)
            out += [f if rel == "." else f"{rel}/{f}".replace(os.sep, "/") for f in files]
        return out

    def find_chirp_file(self) -> str | None:
        """yOCTLoadInterfFromFile_ThorlabsLoadChirp.m l.8-38: any file (recursively) named
        chirp/chrp (case-insensitive) with extension .data or .dat; more than one is an error."""
        if self.exists("data/Chirp.data"):          # fast path (OCITY layout)
            return "data/Chirp.data"
        hits = []
        for f in self.list_files():
            stem, ext = os.path.splitext(f.rsplit("/", 1)[-1])
            if ext.lower() in (".data", ".dat") and stem.lower() in ("chirp", "chrp"):
                hits.append(f)
        if len(hits) > 1:
            raise ValueError(f"Found too many chirp files in {self.dir}: {hits}")
        return hits[0] if hits else None

    def chirp(self, fallback: str | Path | None = None) -> np.ndarray:
        rel = self.find_chirp_file()
        if rel is not None:
            return decode_chirp_bytes(self.read_bytes(rel))
        if fallback is not None:
            return np.loadtxt(fallback, dtype=np.float64).ravel()
        raise FileNotFoundError(f"{self.dir}: no chirp file")

    def lambda_nm(self, oct_system: str, data_dir: str | Path | None = None) -> np.ndarray:
        """Per-pixel wavelength (yOCTLoadInterfFromFile_ThorlabsHeaderLambda.m): chirp from
        the tile, else the chirp file shipped with myOCT for that system."""
        from ..core.geometry import chirp_to_lambda
        sys_l = oct_system.lower()
        fallback = None
        if data_dir is not None and sys_l in _CHIRP_FILE:
            fallback = Path(data_dir) / _CHIRP_FILE[sys_l]
        return chirp_to_lambda(self.chirp(fallback), oct_system)

    def read_bscans(self, file_indices, out: np.ndarray) -> np.ndarray:
        """Read raw Spectral{i}.data files into out[k] (shape (interf, nLambda) int16)."""
        nbytes = self.header.bytes_per_file
        for k, fi in enumerate(file_indices):
            rel = f"data/Spectral{int(fi)}.data"
            if self.layout == "unzipped":
                mv = memoryview(out[k]).cast("B")
                with open(self.dir / rel, "rb", buffering=0) as f:
                    got = f.readinto(mv)
            else:
                b = self.read_bytes(rel)
                got = len(b)
                if got == nbytes:
                    out[k] = np.frombuffer(b, dtype="<i2").reshape(out[k].shape)
            if got != nbytes:
                raise IOError(f"{self.dir}/{rel}: expected {nbytes} bytes, got {got}")
        return out

    def close(self):
        if self._zip is not None:
            self._zip.close()


def open_tile(tile_dir: str | Path) -> TileReader:
    """Alias of TileReader(tile_dir) (backend auto-selected)."""
    return TileReader(tile_dir)


def scan_volume_layout(volume: str | Path, folders) -> dict:
    """Summarise how the tiles of a volume are stored."""
    counts = {"unzipped": 0, "oct": 0, "srr": 0, "wasatch": 0, "simulated": 0, "missing": 0}
    missing = []
    for f in folders:
        lay = tile_layout(Path(volume) / f)
        counts[lay] += 1
        if lay == "missing":
            missing.append(f)
    present = [k for k in ("unzipped", "oct", "srr", "wasatch", "simulated") if counts[k]]
    if not present:
        kind = "missing"
    elif len(present) == 1:
        kind = present[0]
    elif set(present) == {"unzipped", "oct"}:
        kind = "mixed"
    else:
        kind = "mixed_formats"
    return {"layout": kind, "counts": counts, "missing": missing[:20], "n_missing": len(missing)}


def extract_tile(tile_dir: str | Path, delete_archive: bool = False) -> bool:
    """Legacy-equivalent of yOCTUnzipOCTFolder: extract the tile's .oct into the tile
    folder, normalising 'data\\X' entry names to data/X. Returns True if extracted."""
    tile_dir = Path(tile_dir)
    if tile_layout(tile_dir) != "oct":
        return False
    arc = find_oct_archive(tile_dir)
    with zipfile.ZipFile(arc) as z:
        for info in z.infolist():
            name = _norm(info.filename)
            if not name or name.endswith("/"):
                continue
            dest = tile_dir / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            with z.open(info) as src, open(dest, "wb") as dst:
                shutil.copyfileobj(src, dst, 1 << 22)
    ok = (tile_dir / "Header.xml").exists()
    if ok and delete_archive:
        os.remove(arc)
    return ok


def extract_volume(volume: str | Path, folders, delete_archives: bool = False, progress=None) -> dict:
    """Legacy-equivalent of yOCTUnzipTiledScan (sequential; I/O bound)."""
    done = already = 0
    for i, f in enumerate(folders):
        d = Path(volume) / f
        if tile_layout(d) == "unzipped":
            already += 1
        elif extract_tile(d, delete_archives):
            done += 1
        if progress:
            progress(i + 1, len(folders))
    return {"extracted": done, "already_unzipped": already}
