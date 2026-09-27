"""Stage-by-stage comparison of the Python port against the legacy MATLAB
intermediates dumped by validation/matlab/make_reference_plane.m.

Usage: python validation/validate_stages.py [--ref validation/reference_data/ref_y2250.mat] [--device cpu|gpu]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from octrecon.backend import get_xp, to_numpy  # noqa: E402
from octrecon.io.scaninfo import ScanInfo  # noqa: E402
from octrecon.io.thorlabs import read_header, read_chirp, read_bscans_into  # noqa: E402
from octrecon.core.geometry import chirp_to_lambda, build_spectral_geometry, build_tiled_geometry  # noqa: E402
from octrecon.core.spectral import SpectralProcessor  # noqa: E402
from octrecon.core.stitching import OpticalPathCorrection  # noqa: E402


def rel(a, b):
    return float(np.nanmax(np.abs(a - b)) / np.nanmax(np.abs(b)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default=str(ROOT / "validation/reference_data/ref_y2250.mat"))
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--dispersion", type=float, default=8.962e7)
    args = ap.parse_args()
    xp, dev = get_xp(args.device)

    f = h5py.File(args.ref, "r")
    fp = "".join(chr(c) for c in f["D/fp"][()].ravel())
    y_in_file = int(f["yIInFile"][()].ravel()[0])
    tile = Path(fp)
    vol = tile.parent
    print(f"device={dev} tile={tile.name} yIInFile={y_in_file}")

    si = ScanInfo(vol)
    hdr = read_header(tile)
    lam = chirp_to_lambda(read_chirp(tile), si.oct_system)
    sg = build_spectral_geometry(lam, args.dispersion, si.tissue_ri, "sinc5")
    import scipy.io as sio
    focus = sio.loadmat(vol / "zChosenFocusPositions.mat")["focusPositionInImageZpix"].ravel()
    tg = build_tiled_geometry(si, sg.z_um, focus, 2, [-0.03, 0.04])

    raw = np.empty((1, hdr.interf_size, hdr.n_lambda), np.int16)
    read_bscans_into(tile, [y_in_file - 1], raw, hdr)

    for dtype in (np.float64, np.float32):
        sp = SpectralProcessor(sg, hdr.apod_size, xp=xp, dtype=dtype)
        r = xp.asarray(raw)
        x = r.astype(dtype)
        interf = to_numpy(x[:, hdr.apod_size:] - x[:, :hdr.apod_size].mean(1, keepdims=True))[0]
        eq = to_numpy(sp.linearise(r))[0]
        cpx = to_numpy(sp.complex_scan(r))[0]
        mag = to_numpy(sp.magnitude(r))[0]
        print(f"--- {np.dtype(dtype).name}")
        print(f"  apod-subtracted interf  max rel err {rel(interf, f['D/interf'][()]):.2e}")
        print(f"  sinc5 equispaced        max rel err {rel(eq, f['D/interfEquispaced'][()]):.2e}")
        c = f["D/scanCpx"][()]
        print(f"  complex scan            max rel err {rel(cpx, c['real'] + 1j * c['imag']):.2e}")
        print(f"  |scan|                  max rel err {rel(mag, f['D/scanAbs'][()]):.2e}")

    # optical path correction (host, float64)
    oc = OpticalPathCorrection.build(si.optical_path_polynomial(), tg.tile_x_mm, tg.tile_y_mm, tg.tile_z_mm)
    ref_abs = f["D/scanAbs"][()]                         # (nX, nZ)
    j = y_in_file - 1
    r = np.arange(oc.n_rows)
    src = np.clip(r[None, :] + oc.shift[j][:, None], 0, oc.n_rows - 1)
    t = r[None, :] + oc.pdz[j][:, None]
    valid = (t >= 0) & (t <= oc.n_rows - 1)
    opc = np.take_along_axis(ref_abs, src, axis=1) * valid
    ref_valid = f["D/validMap"][()].astype(bool)
    print(f"--- optical path correction (input = MATLAB |scan|)")
    print(f"  valid-map mismatches    {int((valid != ref_valid).sum())} / {valid.size}")
    print(f"  OPC output max abs err  {np.abs(opc - f['D/scanOPC'][()]).max():.2e}")


if __name__ == "__main__":
    main()
