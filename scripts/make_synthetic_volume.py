#!/usr/bin/env python
"""Write a tiny but realistic synthetic tiled OCT scan (for format tests / legacy comparison).

Folder produced (same structure as a real acquisition, e.g. 10um_FOV_1/OCTVolume):
    <out>/ScanInfo.json                 all fields used by octrecon and legacy yOCTProcessTiledScan
    <out>/zChosenFocusPositions.mat     focusPositionInImageZpix (one value per zDepth)
    <out>/synthetic_info.json           generator parameters (focus pixel, reflectors, ...)
    <out>/DataNN/...                    one folder per tile, x-major / z-minor order

Tile formats (--format):
  thorlabs  Header.xml (Thorlabs OCITY structure of a GAN632 tile) + data/Chirp.data (float32)
            + data/OffsetErrors.data + data/Spectral{i}.data (int16 [interfSize, nLambda]);
            --layout oct zips each tile into VolumeGanymedeOCTFile.oct with Windows style
            'data\\SpectralN.data' entry names and removes the unzipped files.
  srr       Data_Y%04d_YTotal%d_B%04d_BTotal%d_<Ganymede|Telesto>.srr + text Chirp.dat
  wasatch   %05d_raw_us_<nLambda>_<nX>_<nBScanAvg>.bin (uint16, no apodization lines)
  wasatch_tif  2D Wasatch: raw_%05d.tif (1-based, one per B-scan repeat, rows = A-lines)
            + raw_bg_00001.tif background (--apod rows; 0 = none); requires --ny 1

Averaging options: --bscan-avg (SpeckleAveraging/SlowAxis, repeated B-scan files),
--ascan-avg (IntensityAveraging/AScans, consecutive repeated A-lines),
--spectra-avg (IntensityAveraging/Spectra = AScanBinning, consecutive raw lines averaged).

Example:
    python scripts/make_synthetic_volume.py /tmp/syn --layout oct --bscan-avg 2
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from octrecon.core.geometry import LAMBDA_RANGE_NM, chirp_to_lambda  # noqa: E402

REAL_CHIRP = Path("/media/atesfet/TS801/Q3_OCT_reconstruction/10um_FOV_1/OCTVolume/Data01/data/Chirp.data")
DATA_DIR = ROOT / "octrecon" / "data"
# real GAN632 probe (10um_FOV_1 ScanInfo.json)
OPTICAL_PATH_POLY = [0.0099168, 0.010817, -0.0001907, -0.00026705, 4.8047e-12]
INSTRUMENT = {  # Model, Series, HardwareConfig, CentralWavelength
    "gan632": ("GAN632", "Ganymede", "GAN632_V1", 880.0),
    "ganymede": ("GAN620C1", "Ganymede", "GAN620C1_V1", 900.0),
    "telesto": ("TEL320C1", "Telesto", "TEL320C1_V1", 1300.0),
}


# --------------------------------------------------------------------------- spectra
def system_chirp(system: str, n_lambda: int) -> np.ndarray:
    s = system.lower()
    if "telesto" in s:
        c = np.loadtxt(DATA_DIR / "ChirpTelesto.dat").ravel()
    elif REAL_CHIRP.exists():
        c = np.fromfile(REAL_CHIRP, "<f4").astype(float)
    else:
        c = np.loadtxt(DATA_DIR / "ChirpGanymede.dat").ravel()
    if len(c) != n_lambda:  # resample the chirp shape to the requested pixel count
        c = np.interp(np.linspace(0, len(c) - 1, n_lambda), np.arange(len(c)), c) * (n_lambda - 1) / (len(c) - 1)
    return c


def wasatch_lambda(n_lambda):
    from octrecon.io.wasatch import wasatch_lambda_nm
    return wasatch_lambda_nm(n_lambda)


class SpectrumModel:
    """Source spectrum x (1 + sum of reflector fringes) + noise, 12-bit counts."""

    def __init__(self, lambda_nm, rng, n_medium=1.33, z_px_air_um=None):
        self.lam = np.asarray(lambda_nm, float)
        self.k = 2 * np.pi / self.lam                                       # 1/nm
        n = len(self.lam)
        p = np.arange(n)
        self.dc = 1400.0 * np.exp(-((p - 0.5 * n) / (0.33 * n)) ** 2) + 250.0
        self.fixed = 4.0 * np.sin(p * 0.37) + 3.0 * rng.standard_normal(n)  # fixed pattern
        self.rng = rng
        self.n_medium = n_medium
        dk = (self.k.max() - self.k.min()) / (n - 1)
        self.z_nyq_um = np.pi / (2 * dk) / 1e3

    def reflectors(self, x_mm, y_mm, z_depth_mm):
        """Air-equivalent depths (um) and amplitudes of the sample reflectors."""
        surf = 480.0 + 40.0 * np.sin(2 * np.pi * x_mm / 0.09) + 25.0 * np.cos(2 * np.pi * y_mm / 0.03)
        surf = surf - z_depth_mm * 1e3 * self.n_medium
        zs = np.array([surf, surf + 45.0, surf + 120.0, surf + 260.0, surf + 400.0])
        amp = np.array([0.20, 0.08, 0.05 * (1 + np.sin(9 * x_mm / 0.09)), 0.04, 0.03])
        return zs, amp

    def aline(self, zs_um, amps):
        s = np.ones_like(self.k)
        for z, a in zip(zs_um, amps):
            s = s + a * np.cos(2 * self.k * z * 1e3 + 0.3)
        return self.dc * s

    def to_counts(self, clean):
        v = clean + self.fixed + 6.0 * self.rng.standard_normal(clean.shape)
        return np.clip(np.round(v), 0, 4095)


def tile_bscans(model, tile, ny, nx, ascan, spectra, bscan, apod, x_px_mm, y_px_mm):
    """-> list over files (y-major, b-minor) of (lines, nLambda) float count arrays,
    lines = apod + nx*ascan*spectra (apod lines first)."""
    xc, yc, zd = tile
    out = []
    apod_clean = model.dc
    for yi in range(ny):
        y_mm = yc + (yi - ny / 2) * y_px_mm
        for _b in range(bscan):
            lines = [model.to_counts(np.broadcast_to(apod_clean, (apod, len(apod_clean))))] if apod else []
            for xi in range(nx):
                x_mm = xc + (xi - nx / 2) * x_px_mm
                zs, amps = model.reflectors(x_mm, y_mm, zd)
                clean = model.aline(zs, amps)
                lines.append(model.to_counts(np.broadcast_to(clean, (ascan * spectra, len(clean)))))
            out.append(np.concatenate(lines, 0))
    return out


# --------------------------------------------------------------------------- writers
def header_xml(system, n_lambda, interf, apod, nx, ny, x_mm, y_mm, ascan, spectra, bscan, n_files):
    model, series, hw, cwl = INSTRUMENT[system]
    df = ('        <DataFile Type="Raw" SizeZ="{n}" SizeX="{i}" RangeZ="1.6558079999999999" RangeX="{xr}" '
          'RangeY="{yr}" BytesPerPixel="2" ApoRegionStart0="0" ApoRegionEnd0="{a}" ScanRegionStart0="{a}" '
          'ScanRegionEnd0="{i}" SizeY="{ny}">data\\Spectral{{k}}.data</DataFile>').format(
        n=n_lambda, i=interf, a=apod, ny=ny, xr=f"{x_mm:g}", yr=f"{y_mm:g}")
    files = "\n".join(df.format(k=k) for k in range(n_files))
    return f"""<?xml version='1.0' encoding='utf-8'?>
<Ocity version="1.0">
    <DataFiles>
{files}
        <DataFile Type="Real" SizeZ="{n_lambda}" RangeZ="1" RangeX="1" RangeY="1" BytesPerPixel="4">data\\Chirp.data</DataFile>
        <DataFile Type="Real" SizeZ="{n_lambda}" RangeZ="1" RangeX="1" RangeY="1" BytesPerPixel="4">data\\OffsetErrors.data</DataFile>
    </DataFiles>
    <Image Type="RawSpectra">
        <SizePixel Unit="px">
            <SizeZ>{n_lambda // 2}</SizeZ>
            <SizeX>{nx}</SizeX>
            <SizeY>{ny}</SizeY>
        </SizePixel>
        <SizeReal Unit="mm">
            <SizeZ>1.655808</SizeZ>
            <SizeX>{x_mm:.6f}</SizeX>
            <SizeY>{y_mm:.6f}</SizeY>
        </SizeReal>
        <PixelSpacing>
            <SizeZ>0.001617</SizeZ>
            <SizeX>{x_mm / nx:.6f}</SizeX>
            <SizeY>{y_mm / ny:.6f}</SizeY>
        </PixelSpacing>
        <DataType>float</DataType>
        <AxisOrder>ZXYT</AxisOrder>
        <CenterX>0.000000</CenterX>
        <CenterY>0.000000</CenterY>
        <Angle>0.000000</Angle>
        <Zoom>1.000000</Zoom>
        <FreeformScanPatternIsActive>False</FreeformScanPatternIsActive>
    </Image>
    <Acquisition>
        <IntensityAveraging>
            <Spectra>{spectra}</Spectra>
            <AScans>{ascan}</AScans>
            <BScans>1</BScans>
        </IntensityAveraging>
        <SpeckleAveraging>
            <FastAxis>1</FastAxis>
            <SlowAxis>{bscan}</SlowAxis>
        </SpeckleAveraging>
        <Timestamp>0</Timestamp>
        <ScanTime Unit="s">0.100000</ScanTime>
        <RefractiveIndex>1.000000</RefractiveIndex>
        <ApodizationType>EachBScan</ApodizationType>
        <ActualSizeOfApodization>{apod}</ActualSizeOfApodization>
        <AcquisitionOrder>FrameByFrame</AcquisitionOrder>
    </Acquisition>
    <Processing>
        <FFTType>NFFT2</FFTType>
        <ApoWindow>Hann</ApoWindow>
        <ApoDataType>AcquiredInScan</ApoDataType>
    </Processing>
    <PolarizationProcessing />
    <Instrument>
        <DevicePresetDescription Category="Speed/Sensitivity">Default (100 kHz A-scan rate)</DevicePresetDescription>
        <Model>{model}</Model>
        <Series>{series}</Series>
        <Serial>SYNTHETIC</Serial>
        <HardwareConfig>{hw}</HardwareConfig>
        <Probe>Probe</Probe>
        <SpectrometerElements>{n_lambda}</SpectrometerElements>
        <BitDepth>12</BitDepth>
        <BytesPerPixel>2</BytesPerPixel>
        <RawDataIsSigned>False</RawDataIsSigned>
        <IsSweptSource>False</IsSweptSource>
        <CentralWavelength>{cwl:.6f}</CentralWavelength>
    </Instrument>
    <MetaInfo>
        <Comment>Synthetic tile written by octrecon scripts/make_synthetic_volume.py</Comment>
    </MetaInfo>
    <MarkerList />
</Ocity>
"""


def write_thorlabs_tile(d: Path, bscans, chirp, system, a):
    (d / "data").mkdir(parents=True, exist_ok=True)
    interf = bscans[0].shape[0]
    (d / "Header.xml").write_text(header_xml(
        system, a.n_lambda, interf, a.apod, a.nx, a.ny, a.nx * a.pixel_um * 1e-3, a.ny * a.pixel_um * 1e-3,
        a.ascan_avg, a.spectra_avg, a.bscan_avg, len(bscans)), encoding="utf-8")
    chirp.astype("<f4").tofile(d / "data" / "Chirp.data")
    np.zeros(a.n_lambda, "<f4").tofile(d / "data" / "OffsetErrors.data")
    for k, b in enumerate(bscans):
        b.astype("<i2").tofile(d / "data" / f"Spectral{k}.data")


def zip_tile(d: Path):
    """Thorlabs .oct = ZIP with Windows separators; remove the unzipped files."""
    arc = d / "VolumeGanymedeOCTFile.oct"
    files = [d / "Header.xml"] + sorted((d / "data").iterdir())
    with zipfile.ZipFile(arc, "w", zipfile.ZIP_STORED) as z:
        for f in files:
            name = "Header.xml" if f.name == "Header.xml" else "data\\" + f.name
            z.writestr(zipfile.ZipInfo(name, date_time=(2026, 9, 27, 0, 0, 0)), f.read_bytes())
    (d / "Header.xml").unlink()
    shutil.rmtree(d / "data")


def write_srr_tile(d: Path, bscans, chirp, system, a, rng):
    d.mkdir(parents=True, exist_ok=True)
    sysname = "Telesto" if "telesto" in system else "Ganymede"
    np.savetxt(d / "Chirp.dat", chirp, fmt="%.6f")
    lines = bscans[0].shape[0]
    txt = (f"size1={a.n_lambda} size2={lines} bitdepth=12 "
           f"aporegions=1,0,{a.apod} scanregions=1,{a.apod},{lines} ").encode()
    for k, b in enumerate(bscans):
        y0, b0 = divmod(k, a.bscan_avg)
        name = "Data_Y%04d_YTotal%d_B%04d_BTotal%d_%s.srr" % (y0 + 1, a.ny, b0 + 1, a.bscan_avg, sysname)
        data = b.astype(np.uint16)
        # upper 4 bits are not data (legacy rem(x,4096)): set some to test masking
        data = data + (rng.integers(0, 16, data.shape).astype(np.uint16) << 12)
        with open(d / name, "wb") as f:
            f.write(b"headersize=%d\n" % len(txt))
            f.write(txt)
            f.write(data.astype("<u2").tobytes())


def write_wasatch_tif_tile(d: Path, bscans, a):
    import tifffile
    d.mkdir(parents=True, exist_ok=True)
    for k, b in enumerate(bscans):
        tifffile.imwrite(d / ("raw_%05d.tif" % (k + 1)), b[a.apod:].astype(np.uint16))
    if a.apod:
        tifffile.imwrite(d / "raw_bg_00001.tif", bscans[0][:a.apod].astype(np.uint16))


def write_wasatch_tile(d: Path, bscans, a):
    d.mkdir(parents=True, exist_ok=True)
    for k, b in enumerate(bscans):
        b.astype("<u2").tofile(d / ("%05d_raw_us_%d_%d_%d.bin" % (k, a.n_lambda, b.shape[0], a.bscan_avg)))


# --------------------------------------------------------------------------- main
def make_volume(out, layout="unzipped", fmt="thorlabs", system="gan632", bscan_avg=1, ascan_avg=1,
                spectra_avg=1, nx=40, ny=6, tiles_x=2, tiles_y=1, depths=(0.0, 0.01), pixel_um=2.0,
                n_lambda=2048, apod=25, seed=0, focus_pix=None, write_octsystem=True, overwrite=True):
    a = argparse.Namespace(**locals())
    out = Path(out)
    if out.exists():
        if not overwrite:
            raise FileExistsError(out)
        shutil.rmtree(out)
    out.mkdir(parents=True)
    rng = np.random.default_rng(seed)
    if fmt != "thorlabs" and (ascan_avg != 1 or spectra_avg != 1):
        raise ValueError("A-scan averaging / spectral binning only exist in the Thorlabs OCITY format")
    if fmt == "wasatch_tif" and ny != 1:
        raise ValueError("2D Wasatch (tif) scans have a single y frame: use ny=1")
    if fmt in ("wasatch", "wasatch_tif"):
        lam, chirp = wasatch_lambda(n_lambda), None
        if fmt == "wasatch":
            a.apod = 0
        oct_system = "Wasatch"
    else:
        sys_for_lambda = system if fmt == "thorlabs" else ("telesto_srr" if "telesto" in system else "ganymede_srr")
        chirp = system_chirp(system, n_lambda)
        lam = chirp_to_lambda(chirp, sys_for_lambda)
        oct_system = {"gan632": "gan632", "ganymede": "Ganymede", "telesto": "Telesto"}[system] \
            if fmt == "thorlabs" else ("Telesto_SRR" if "telesto" in system else "Ganymede_SRR")
    model = SpectrumModel(lam, rng)
    tile_x_mm, tile_y_mm = nx * pixel_um * 1e-3, ny * pixel_um * 1e-3
    xc = (np.arange(tiles_x) - (tiles_x - 1) / 2) * tile_x_mm
    yc = (np.arange(tiles_y) - (tiles_y - 1) / 2) * tile_y_mm
    zd = np.asarray(depths, float)
    folders, gx, gy, gz = [], [], [], []
    k = 1
    for y in yc:                       # legacy yOCTScanTile order: y, then x, then z
        for x in xc:
            for z in zd:
                folders.append(f"Data{k:02d}")
                gx.append(float(x)); gy.append(float(y)); gz.append(float(z))
                k += 1
    for f, x, y, z in zip(folders, gx, gy, gz):
        bs = tile_bscans(model, (x, y, z), ny, nx, ascan_avg, spectra_avg if fmt == "thorlabs" else 1,
                         bscan_avg, a.apod, pixel_um * 1e-3, pixel_um * 1e-3)
        d = out / f
        if fmt == "thorlabs":
            write_thorlabs_tile(d, bs, chirp, system, a)
            if layout == "oct":
                zip_tile(d)
        elif fmt == "srr":
            write_srr_tile(d, bs, chirp, system, a, rng)
        elif fmt == "wasatch_tif":
            write_wasatch_tif_tile(d, bs, a)
        else:
            write_wasatch_tile(d, bs, a)

    if focus_pix is None:  # surface ~ 480 um air-equivalent -> pixel in the tissue z grid
        z_air_px_um = model.z_nyq_um / (n_lambda / 2)
        focus_pix = int(round(480.0 / z_air_px_um)) + 1
    focus = [float(focus_pix)] * len(zd)
    si = {
        "isVerifyMotionRange": True, "nBScanAvg": bscan_avg, "octProbeFOV_mm": 1,
        "octProbePath": "synthetic\\Probe Olympus - 20x - OCTG - WINTER.ini",
        "pixelSize_um": pixel_um, "tissueRefractiveIndex": 1.33, "unzipOCTFile": layout != "oct",
        "xOffset": 0, "xRange_mm": [float(xc.min() - tile_x_mm / 2), float(xc.max() + tile_x_mm / 2)],
        "yOffset": 0, "yRange_mm": [float(yc.min() - tile_y_mm / 2), float(yc.max() + tile_y_mm / 2)],
        "zDepths": [float(v) for v in zd], "units": "mm", "version": 1.1,
        "octSystem": oct_system,
        "octProbe": {"ObjectiveName": "Olympus20xOCTGWinter", "ObjectiveWorkingDistance": 3.5,
                     "DefaultDispersionQuadraticTerm": 8.6883e7,
                     "OpticalPathCorrectionPolynomial": OPTICAL_PATH_POLY,
                     "GalvoPhaseDelay_Asamples": 19.15},
        "xCenters_mm": [float(v) for v in xc], "yCenters_mm": [float(v) for v in yc],
        "tileRangeX_mm": tile_x_mm, "tileRangeY_mm": tile_y_mm,
        "galvoPhaseDelayXOffsetCorrection_mm": 0.0,
        "gridXcc": gx, "gridZcc": gz, "gridYcc": gy,
        "scanOrder": list(range(1, len(folders) + 1)), "octFolders": folders,
        "nXPixelsInEachTile": nx, "nYPixelsInEachTile": ny,
    }
    if not write_octsystem:
        del si["octSystem"]
    (out / "ScanInfo.json").write_text(json.dumps(si, indent=1))
    import scipy.io as sio
    sio.savemat(out / "zChosenFocusPositions.mat", {"focusPositionInImageZpix": np.asarray(focus)[None, :]})
    info = {"format": fmt, "layout": layout, "system": system, "oct_system": oct_system,
            "bscan_avg": bscan_avg, "ascan_avg": ascan_avg, "spectra_avg": spectra_avg,
            "nx": nx, "ny": ny, "tiles": [tiles_x, tiles_y, len(zd)], "n_lambda": n_lambda,
            "apod": a.apod, "focus_pix": focus_pix, "seed": seed, "pixel_um": pixel_um,
            "suggested_dispersion_quadratic_term": 0.0}
    (out / "synthetic_info.json").write_text(json.dumps(info, indent=1))
    return info


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out")
    ap.add_argument("--layout", choices=["unzipped", "oct"], default="unzipped")
    ap.add_argument("--format", dest="fmt", choices=["thorlabs", "srr", "wasatch", "wasatch_tif"], default="thorlabs")
    ap.add_argument("--system", choices=["gan632", "ganymede", "telesto"], default="gan632")
    ap.add_argument("--bscan-avg", type=int, default=1)
    ap.add_argument("--ascan-avg", type=int, default=1)
    ap.add_argument("--spectra-avg", type=int, default=1, help="AScanBinning")
    ap.add_argument("--nx", type=int, default=40)
    ap.add_argument("--ny", type=int, default=6)
    ap.add_argument("--tiles-x", type=int, default=2)
    ap.add_argument("--tiles-y", type=int, default=1)
    ap.add_argument("--depths", default="0,0.01", help="comma separated zDepths (mm)")
    ap.add_argument("--pixel-um", type=float, default=2.0)
    ap.add_argument("--n-lambda", type=int, default=2048)
    ap.add_argument("--apod", type=int, default=25, help="apodization A-lines per file (Thorlabs/SRR)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-octsystem", action="store_true", help="omit octSystem from ScanInfo.json")
    a = ap.parse_args(argv)
    if a.layout == "oct" and a.fmt != "thorlabs":
        ap.error("--layout oct only applies to --format thorlabs")
    info = make_volume(a.out, a.layout, a.fmt, a.system, a.bscan_avg, a.ascan_avg, a.spectra_avg, a.nx, a.ny,
                       a.tiles_x, a.tiles_y, [float(v) for v in a.depths.split(",")], a.pixel_um,
                       a.n_lambda, a.apod, a.seed, write_octsystem=not a.no_octsystem)
    print(json.dumps(info))


if __name__ == "__main__":
    main()
