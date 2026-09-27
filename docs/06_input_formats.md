# 06 — Supported raw input formats

The port accepts every raw input that the legacy tiled reconstruction
(`yOCTProcessTiledScan`) accepts, plus a few that the legacy code *intends* to accept but
cannot process in a tiled scan because of bugs. All of them are validated against the
**unmodified** legacy MATLAB code (R2024a) on small synthetic scans (§6).

Legacy paths below are relative to `Reconstruction_code_legacy/myOCT/`.

## 1. What a tiled scan is

A tiled scan is always a folder holding `ScanInfo.json` (grid, tile folders, probe, system)
and one sub-folder per tile (`octFolders`, e.g. `Data01 … Data864`). Only the content of
the tile folders varies between formats; `ScanInfo.json` handling is identical
(`octrecon/io/scaninfo.py`). You may point the tool at the volume folder or at its parent
containing `OCTVolume/`.

Every tile folder is opened with `TileReader(tile_dir)` (`octrecon/io/volume.py`), which
picks a backend from the folder content and returns the same interface for all formats:

| member | meaning |
|---|---|
| `header` | `TileHeader`: `n_lambda`, `interf_size` (A-lines per raw file incl. apodization), `apod_size`, `size_x`, `size_y`, `ascan_avg`, `bscan_avg`, `spectra_avg` (AScanBinning), `raw_dtype`, `apod_mode` (`lines` / `frame_mean`), `manufacturer`, `bytes_per_file` |
| `lambda_nm(octSystem, data_dir)` | per-pixel wavelength (chirp + system range, or polynomial / stored values) |
| `read_bscans(file_indices, out)` | `out[k] = (interf_size, n_lambda)` raw spectra of file k; file index = `(y-1)*nBScanAvg + (b-1)` (0-based, like `Spectral{i}.data`) |

`SpectralProcessor` (`octrecon/core/spectral.py`) casts to float, applies AScanBinning /
frame-mean background when needed, subtracts the apodization and runs the validated
sinc5 → window/dispersion → IFFT chain; `Reconstructor.process_row` averages |scan| over
B-scan and A-scan repeats. The same code runs on CPU (numba) and GPU (CuPy; the fused CUDA
kernel is compiled for int16, uint16 and float32 raw input). Note: the format validation
below ran on CPU only (the GPU driver was unavailable); the GPU path shares the same
preparation code and was previously validated for int16 OCITY data.

## 2. Formats

| # | Tile content | `TileReader` backend / layout | octSystem names | Legacy tiled processing |
|---|---|---|---|---|
| A | `Header.xml` + `data/Spectral{i}.data`, `data/Chirp.data` (Thorlabs OCITY, unzipped) | `TileReader`, `unzipped` | `Ganymede`, `Telesto`, `Gan632` (case-insensitive) | yes |
| B | `VolumeGanymedeOCTFile.oct` (or any single `*.oct`): ZIP of A, entry names `data\Spectral0.data` | `TileReader`, `oct` | same as A | yes (unzips with 7-Zip, deletes the archive) |
| C | `Data_Y%04d_YTotal%d_B%04d_BTotal%d_<Ganymede\|Telesto>.srr` (+ text `Chirp.dat`) | `SRRTileReader` (`io/srr.py`), `srr` | `Ganymede_SRR`, `Telesto_SRR` | **no** — legacy bug (§4) |
| D | `%05d_raw_us_<nLambda>_<nX>_<nBScanAvg>.bin` (3D) or `raw_%05d.tif` + `raw_bg_00001.tif` (2D) | `WasatchTileReader` (`io/wasatch.py`), `wasatch` | `Wasatch` | **no** — legacy bug (§4) |
| E | `data.mat` (`interf`, `dim`) from `Simulation/yOCTSimulateTileScan.m` | `SimulatedTileReader` (`io/simulated.py`), `simulated` | `Simulated Ganymede` | yes |

Detection of the backend (`tile_layout`) looks only at file names: `Header.xml`+`data/` →
A; a `.oct` archive → B; `Data_Y*.srr` → C; Wasatch `.bin`/`raw_0*.tif` → D; `data.mat` → E.
The octSystem name is then mapped to a manufacturer exactly like
`LoadSave/yOCTLoadInterfFromFile.m` l.104-117 and must match the files found
(e.g. `Ganymede` with `.srr` files is rejected with a clear error instead of failing later).

### A. Thorlabs OCITY, unzipped (reference format)

* Header: `LoadSave/yOCTLoadInterfFromFile_ThorlabsHeader.m`. Used fields: first
  `DataFile` whose text starts with `data\Spectral` → `SizeZ` (nLambda), `SizeX`
  (interfSize), `ApoRegionEnd0` (apodSize) (l.165-179); `Image/SizePixel/SizeX|SizeY`,
  `Image/SizeReal/SizeY` (l.85-132: `SizeReal/SizeY` missing or 0 → single y frame, negative
  → the y axis is the B-scan-average axis); `Acquisition/IntensityAveraging/AScans`
  (A-scan averaging, l.135-143), `.../Spectra` (AScanBinning, l.182),
  `Acquisition/SpeckleAveraging/SlowAxis` (B-scan averaging, l.145-161).
* Chirp: `yOCTLoadInterfFromFile_ThorlabsLoadChirp.m` — any file (recursively) named
  `chirp`/`chrp` with extension `.data`/`.dat`; parsed as text first, binary float32 if that
  yields < 2 numbers (l.52-62); if there is none, the chirp shipped with myOCT
  (`octrecon/data/Chirp{Ganymede,Telesto}.dat`, `ThorlabsHeaderLambda.m` l.27-37).
  λ = 1/(chirp·(1/λmax−1/λmin)/(N−1) + 1/λmin) with the per-system range of
  `ThorlabsHeaderLambda.m` l.9-24 (Ganymede/Ganymede_SRR 796.23–1010.02 nm, Gan632 796–1010 nm,
  Telesto/Telesto_SRR 1208.69–1372.50 nm). As in legacy, all tiles use the first tile's
  header and chirp (`dimOneTile`).
* Data: `yOCTLoadInterfFromFile_ThorlabsData.m`. `Spectral{i}.data` = int16
  `[interfSize, nLambda]`; file index `(y-1)*nBScanAvg + b-1` (l.71-81); first `apodSize`
  A-lines are the apodization, the mean of which is subtracted
  (`yOCTLoadInterfFromFile.m` l.201-216).
* **AScanBinning** (`Spectra` > 1), l.120-127: `filter2(ones(1,B)/B, interf)` ('same',
  zero padded) along the A-line axis then keep columns `max(1,floor(B/2)):B:end`.
  Ported exactly in `core/spectral.py:bin_ascans` — including the edge effect for odd B
  (the first kept column of B=3 averages a zero pad, e.g. `(0+x1+x2)/3`). The apodization
  lines are not binned. The pipeline checks that the kept A-line count equals
  `SizeX × AScans` (legacy would fail in the reshape).
* **A-scan averaging** (`AScans` > 1), l.131-137: A-lines are grouped as
  `reshape([nLambda, AScanAvg, SizeX])`, i.e. the repeats of one x position are
  consecutive. Legacy averages the **magnitude** (`yOCTProcessTiledScan.m` l.288-291:
  `abs`, then `mean` over the trailing dims, B-scan repeats first, then A-scan repeats);
  `process_row` does the same.
* **B-scan averaging** (`SlowAxis` > 1): repeats are consecutive files, |scan| averaged.
* `dimensions.x.values` / `y.values` come from `ScanInfo.json`
  (`yOCTProcessTiledScan_createDimStructure.m` l.53-57), while the number of A-scans per
  frame comes from the header; the port checks `Header SizeX == nXPixelsInEachTile`
  (legacy `interp2` would fail otherwise).

### B. Thorlabs `.oct` archive

`VolumeGanymedeOCTFile.oct` is a ZIP of A with Windows separators in entry names
(`data\Spectral0.data`). Legacy (`yOCTProcessTiledScan.m` l.110-112 →
`LoadSave/yOCTUnzipTiledScan.m` l.54-100 → `yOCTUnzipOCTFolder.m`) extracts every tile
with the external `7z` tool (l.77-91; on Linux p7zip writes files literally named
`data\Spectral0.data`, which the "malformed file name" fix-up l.98-138 moves into `data/`),
then **deletes the archive** (l.165-181). The port offers:

* `raw_input: auto` (default): read the archive **in place** with Python's `zipfile`
  (no extraction, no extra disk; entry names normalised; reads serialised per tile).
* `raw_input: extract`: extract like legacy (`io/volume.py:extract_volume`), keeping the
  archive unless `delete_archives_after_extract: true`.
* Mixed volumes (some tiles unzipped, some not) are fine in both modes.

### C. Thorlabs SRR (`Ganymede_SRR`, `Telesto_SRR`)

`LoadSave/yOCTLoadInterfFromFile_ThorlabsSRRHeader.m`, `_ThorlabsSRRData.m`. One file per
B-scan repeat named `Data_Y%04d_YTotal%d_B%04d_BTotal%d_<System>.srr`: line 1
`headersize=<n>`, then `n` bytes of `key=value` tokens (`size1` = nLambda,
`scanregions=k,start,end`, `aporegions=k,start,end`, 0-based start / exclusive end), then
uint16 samples, A-line major, of which only the low 12 bits are data (`rem(x,4096)`,
SRRData l.104-124). The file must contain exactly `size1*scanend` samples (legacy
`_ReadFile` size check). nY = `YTotal`, nBScanAvg = `BTotal` from the first file name.
The reader returns int16 arrays with the apodization A-lines first (same layout as A).
Chirp: text file in the folder, else myOCT's packaged chirp.

### D. Wasatch (`Wasatch`)

`LoadSave/yOCTLoadInterfFromFile_WasatchHeader.m`, `_WasatchData.m`.
* 3D: `%05d_raw_us_<nLambda>_<nX>_<nBScanAvg>.bin`, uint16 `reshape(nLambda, nX)`, file
  index 0-based `(y-1)*nBScanAvg + b-1`, nY = nFiles/nBScanAvg. There are **no
  apodization lines**: legacy tries `raw_bg_00001.tif` with the `.bin` reader, which always
  fails, and falls back to the mean over all A-lines of the frames loaded together
  (l.115-131; then `yOCTLoadInterfFromFile.m` l.211 averages over the B-scan repeats). In the
  tiled pipeline one y frame (with its repeats) is loaded at a time, so the background is
  the mean over x and repeats of that frame — `apod_mode: frame_mean`.
* 2D: `raw_%05d.tif` (1-based; rows = A-lines), nBScanAvg = number of files, nY = 1,
  background = mean of the rows of `raw_bg_00001.tif` if present, else frame mean.
* λ(nm) = 657.328 + 0.0828908·p − 1.10175e-6·p² − 7.04714e-11·p³, p = 1..nLambda
  (WasatchHeader l.66-73).

### E. Simulated (`Simulated Ganymede`)

Tiles written by `Simulation/yOCTSimulateTileScan.m` (l.82, l.107-118): `data.mat` with
`interf` (single, `(λ, x, y[, AScanAvg, BScanAvg])`) and `dim` (λ in nm). Legacy loads the
header from `dim` (`yOCTLoadInterfFromFile.m` l.131-132) and subtracts a zero apodization
(`_SimulatedData.m`). MATLAB v7 (`scipy.io`) and v7.3 (`h5py`) files are read.

## 3. System auto-detection

`octrecon/io/detect.py:what_oct_system_is_it` ports
`LoadSave/yOCTLoadInterfFromFile_WhatOCTSystemIsIt.m`:

1. list files (recursively) with extension `.xml` (Thorlabs), `.dat` (SRR text chirp),
   `.bin`/`.tif` (Wasatch) (l.24-29);
2. drop every `.tif` whose name does not match `textscan(fname,'%sraw_%d')` (l.37-51) —
   since `%s` swallows the whole name this drops **all** `.tif` files (verified in
   MATLAB), so 2D Wasatch folders are never detected; replicated;
3. more than one manufacturer → error (l.58-60);
4. Thorlabs: `Header.xml` must contain exactly one of `Ganymed` / `Telesto`
   (l.77-99). **A GAN632 header contains `<Series>Ganymede</Series>` and is detected as
   `Ganymede`** (λ range 796.23–1010.02 nm instead of Gan632's 796–1010 nm: 0.13 % larger
   depth pixel, ≈1.3 px over a 1024-px A-scan) — legacy behaviour, kept; write `octSystem: gan632` in
   ScanInfo.json to get the Gan632 range. SRR: the single `Data_Y0001_*B0001*.srr` must
   name exactly one of Ganymede/Telesto (l.101-125).

It is used when `ScanInfo.json` has neither `octSystem` nor the deprecated `OCTSystem`
(checked on the first tile; `ScanInfo.oct_system_source` records it). **Legacy
`yOCTProcessTiledScan` errors in that case** (l.141-148); the port falls back to the rule
single-scan loading uses. Extensions over legacy detection: a tile holding only a `.oct`
archive is identified from the `Header.xml` inside it, a tile holding only `data.mat` is
`Simulated Ganymede`, and the `contains(path, ext)` tests are applied to the path relative
to the tile folder (legacy tests the absolute path, so a parent folder named e.g. `x.dat`
would confuse it).

## 4. Legacy behaviours intentionally not replicated

| Legacy behaviour | Port | Why |
|---|---|---|
| SRR tiled scans fail: `yOCTLoadInterfFromFile.m` l.127-129 calls `_ThorlabsSRRHeader` and then **overwrites** the result with `_WasatchHeader`, which errors ("Cannot find … `*_raw_us_*.bin`") | SRR header used as intended | obvious bug; verified by running legacy on a synthetic SRR scan |
| Wasatch tiled scans fail: no Wasatch branch in the header `switch` (l.123-133), so `dimensions` only has `aux` → "Unrecognized field name lambda/y" | Wasatch header ported | same |
| SRR header without an explicit system takes `Telesto`/`Ganymede` (no `_SRR`) from the file name (`_ThorlabsSRRHeader.m` l.50-52), which then selects the non-SRR data reader | the file-name system + `_SRR` | consistent with WhatOCTSystemIsIt |
| Missing `octSystem` in ScanInfo.json → error | auto-detection (§3) | convenience; identical result to naming the detected system |
| Missing / truncated / unreadable raw file → warning, frame replaced by NaN (`yOCTLoadInterfFromFile_ReadFile.m`) | `IOError` naming the file | silent NaN planes in a 1 TB volume are worse than a clear error; re-run after fixing the file |
| `.oct` archives extracted with 7-Zip and deleted | read in place by default; extraction optional, archive kept by default | no extra disk / no destructive step |

## 5. Not supported

| Input | Reason |
|---|---|
| AWS `s3://` input/output paths (`awsModifyPathForCompetability`, `awsSetCredentials`, AWS CLI) | out of scope for a local GPU workstation tool; the MATLAB code streams individual files through `imageDatastore`. Copy or mount (e.g. `s3fs`/`mount-s3`) the volume locally — mounted paths work like any folder. |
| Thorlabs 1D mode (`data/SpectralFloat.data`, `ThorlabsData.m` l.39-63) | point scans (sizeX = 1) are not tiled volumes: `yOCTProcessTiledScan` needs an x/y grid per tile |
| Mixing formats (e.g. SRR and OCITY tiles) in one volume | legacy uses one system for all tiles; the port reports `layout: mixed_formats` and uses each tile's own reader but the first tile's header/wavelengths, so only mix storage layouts A/B |
| `interpMethod: pchip` | not a format question; `core/geometry.py` supports sinc interpolation only |

## 6. Validation against the unmodified legacy code

Synthetic scans from `scripts/make_synthetic_volume.py` (2×1 tiles × 2 depths, 40×6 px
tiles, 2 µm pixels, 2048 spectral pixels, 25 apodization lines, reflectors + noise +
fixed pattern, real Data01 chirp, real GAN632 optical-path polynomial; `--layout`,
`--format`, `--bscan-avg`, `--ascan-avg`, `--spectra-avg`, `--system` options).
Legacy: `validation/matlab/run_legacy_synthetic.m` (full `yOCTProcessTiledScan`, 2-worker
pool, p7zip for `.oct`) and `dump_synthetic_plane.m` (serial copy of its plane loop, float
dB). Parameters: dispersion 8.949e7, focusSigma 10, focus pixel from the generator,
outputFilePixelSize_um = pixel size, sinc5, no crop. Python: `ReconConfig(device='cpu',
precision='float64', legacy_double_quantization=True)`.

| Variant | float dB, all 6 planes: max \|Δ\| (float64) | NaN/finite mismatches | final TIFF: max \|Δ\| |
|---|---|---|---|
| unzipped (baseline, gan632) | 1.9e-6 dB | 570 (all weight ties, see below) | 1 LSB (1.2e-3 dB) |
| `.oct` read in place (vs legacy 7-Zip extraction) | 1.9e-6 dB, output identical to baseline | 570 | 1 LSB |
| `.oct` extracted by the port | 1.9e-6 dB, identical to baseline | 570 | 1 LSB |
| B-scan averaging 2 | 1.8e-6 dB | 570 | 1 LSB |
| A-scan averaging 2 | 1.9e-6 dB | 570 | 1 LSB |
| AScanBinning 2 | 1.9e-6 dB | 570 | 1 LSB |
| AScanBinning 3 (odd, edge effect) | 1.9e-6 dB | 570 | 1 LSB |
| B-scan 2 + A-scan 2 + binning 2, `.oct` | 1.7e-6 dB | 570 | 1 LSB |
| system `Ganymede` | 1.9e-6 dB | 456 | 1 LSB |
| system `Telesto` (packaged Telesto chirp) | 1.9e-6 dB | 252 | 1 LSB |
| `Simulated Ganymede` (legacy simulator output, 4 planes) | 3.3e-5 dB within 60 dB of peak (1.7e-4 within 80 dB) | 0 | — |
| SRR, Wasatch 3D, no `octSystem` | legacy fails (§4); port reconstructs | — | — |
| real 10um_FOV_1 plane y=2250 (regression) | 9.5e-7 dB | 0 | — |

The float differences are the float32 rounding of the output. **All** NaN/finite
mismatches are output pixels whose legacy total weight is exactly `exp(-4.5)` (to 1e-12):
pixels far (> 3σ) from every focus that receive only the floor weight of a single tile;
`weight < exp(-4.5) → NaN` then depends on the last bit of the interpolation weights
(`core/stitching.py`, not a format issue; in the real run the crop range excludes them).
TIFF differences of 1 LSB come from these pixels changing a plane's min/max and hence the
legacy double quantisation. The simulated scan is noiseless: its background (~140 dB below
the peak) is round-off noise and legacy computes it partly in single precision
(`interf` is stored as single), so only the signal is comparable.

Loader level (`validation/matlab/dump_legacy_loader.m`, stored in
`tests/data/legacy_formats_ref.npz`): the apodization-corrected interferogram of one y
frame from our readers equals `yOCTLoadInterfFromFile` output to ≤ 1e-9 (exact for
integer data) for Thorlabs (B-scan 2 × A-scan 2 × binning 3; Telesto binning 2), SRR
(Ganymede B-scan 2, Telesto), Wasatch 3D (B-scan 2), Wasatch 2D with and without
background; raw apodization lines are bit-identical; wavelengths agree to 1e-12;
`WhatOCTSystemIsIt` results agree for all six folders (incl. the legacy error for 2D
Wasatch).

`tests/test_formats.py` (32 tests, ~6 s, no MATLAB needed) regenerates the synthetic inputs
(checksummed) and checks all of the above against the stored references, plus
unzipped ≡ `.oct` in place ≡ extracted (bit-identical), a full run to TIFF from `.oct`,
auto-detection, `.oct`/`data.mat` detection, SRR/Wasatch tiled volumes (with and without
`octSystem`), SRR upper-bit masking, system/format mismatch errors and the `filter2`
edge behaviour. `python tests/test_formats.py --build-reference` rebuilds the references
(needs MATLAB, ~3 min).
