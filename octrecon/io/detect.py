"""OCT system / manufacturer identification.

`what_oct_system_is_it(folder)` ports LoadSave/yOCTLoadInterfFromFile_WhatOCTSystemIsIt.m
(used by yOCTLoadInterfFromFile when no 'OCTSystem' is given):
  1. list files (recursively) with extension .xml (Thorlabs Header.xml), .dat (Thorlabs
     SRR text chirp), .bin / .tif (Wasatch). Legacy then drops every .tif whose name does
     not match textscan(fname,'%sraw_%d') - since '%s' swallows the whole name this drops
     ALL .tif files (verified in MATLAB R2024a), so 2D Wasatch tif folders are never
     detected; we replicate that.
  2. more than one manufacturer -> error; none -> error.
  3. Thorlabs: Header.xml text contains exactly one of 'Ganymed' / 'Telesto'
     -> 'Ganymede' / 'Telesto' (a GAN632 header says <Series>Ganymede</Series>, so it is
     detected as 'Ganymede' - legacy behaviour; set octSystem='gan632' explicitly to get
     the Gan632 wavelength range).
     SRR: the single file Data_Y0001_*B0001*.srr; its name contains exactly one of
     'Ganymede' / 'Telesto' -> 'Ganymede_SRR' / 'Telesto_SRR'. Wasatch -> 'Wasatch'.
Extensions (not in legacy): a tile holding only a Thorlabs .oct archive is identified from
the Header.xml inside the archive (legacy unzips before it ever detects); a tile holding
only data.mat (yOCTSimulateTileScan) -> 'Simulated Ganymede' (legacy errors).

`manufacturer_of(system)` ports the name -> manufacturer switch of
yOCTLoadInterfFromFile.m l.104-117.
"""
from __future__ import annotations

import fnmatch
import os
import zipfile
from pathlib import Path

from .thorlabs import detect_thorlabs_system

SYSTEM_MANUFACTURER = {"ganymede": "Thorlabs", "telesto": "Thorlabs", "gan632": "Thorlabs",
                       "ganymede_srr": "Thorlabs_SRR", "telesto_srr": "Thorlabs_SRR",
                       "wasatch": "Wasatch", "simulated ganymede": "Simulated"}


def manufacturer_of(oct_system: str) -> str:
    m = SYSTEM_MANUFACTURER.get(str(oct_system).lower())
    if m is None:
        raise ValueError(f'Wrong OCTSystem name! Received: "{oct_system}". Expected: Ganymede, Telesto, '
                         "Gan632, Ganymede_SRR, Telesto_SRR, Wasatch or Simulated Ganymede")
    return m


def _walk(folder: Path):
    for root, _, files in os.walk(folder):
        rel = os.path.relpath(root, folder)
        for f in files:
            yield f if rel == "." else os.path.join(rel, f).replace(os.sep, "/")


def what_oct_system_is_it(folder: str | Path) -> tuple[str, str]:
    """-> (octSystem, manufacturer) for one scan (tile) folder."""
    folder = Path(folder)
    if not folder.is_dir():
        raise FileNotFoundError(f'"{folder}" doesn\'t exist, mistake?')
    files = [f for f in _walk(folder) if os.path.splitext(f)[1].lower() in (".xml", ".dat", ".bin", ".tif")]
    files = [f for f in files if os.path.splitext(f)[1].lower() != ".tif"]     # legacy textscan quirk
    # legacy tests contains(fullpath, ext); we test the path relative to the scan folder
    is_thor = any(".xml" in f for f in files)
    is_srr = any(".dat" in f for f in files)
    is_was = any(".bin" in f or ".tif" in f for f in files)
    if not (is_thor or is_srr or is_was):
        from .volume import find_oct_archive
        if (folder / "data.mat").exists():
            return "Simulated Ganymede", "Simulated"
        arc = find_oct_archive(folder)
        if arc is not None:
            with zipfile.ZipFile(arc) as z:
                names = {n.replace("\\", "/").lstrip("/"): n for n in z.namelist()}
                if "Header.xml" in names:
                    return detect_thorlabs_system(z.read(names["Header.xml"])), "Thorlabs"
        raise ValueError(f"Couldn't figure out what is the manufacturer of the OCT system in {folder}")
    if is_thor + is_srr + is_was > 1:
        raise ValueError(f"Couldn't determine OCT system, there are multiple manufacturers in this folder: {folder}")
    if is_was:
        return "Wasatch", "Wasatch"
    if is_thor:
        return detect_thorlabs_system((folder / "Header.xml").read_bytes()), "Thorlabs"
    # SRR
    first = [p.name for p in folder.iterdir() if fnmatch.fnmatchcase(p.name, "Data_Y0001_*B0001*.srr")]
    if len(first) != 1:
        raise ValueError("Expected only one first file in dataset, does this folder contain more than one OCT scan?")
    g, t = "Ganymede" in first[0], "Telesto" in first[0]
    if g and not t:
        return "Ganymede_SRR", "Thorlabs_SRR"
    if t and not g:
        return "Telesto_SRR", "Thorlabs_SRR"
    raise ValueError("Cannot determine OCT system SRR")
