# 01 — Legacy MATLAB reconstruction (myOCT): how it works

Source analysed: `Reconstruction_code_legacy/myOCT` (MATLAB, Yolab "myOCT").
`Scanning code/myOCT-master_8.27.26/myOCT-master_8.2.26` is the copy that *acquired*
the data; `diff -rq` shows its `Processing/`, `LoadSave/`, `Utils/`, `ThorlabsImager/`
are **byte-identical** to the legacy copy. The legacy copy additionally contains
`yOCTMeasureFocusDrift*.m` (which produced `zChosenFocusPositions.mat`) and the only
demo with an active processing block, so **`Reconstruction_code_legacy/myOCT` is the
authoritative reconstruction code** for this dataset. (The ADLZ_1 path mentioned in
the brief, `/media/atesfet/ADLZ_1/.../Demo_ScanAndProcess3D.m`, is not mounted here;
`Reconstruction_code_legacy/myOCT/Demo_ScanAndProcess3D.m` is that script.)

Everything below was verified by running the legacy code in MATLAB R2024a
(`validation/matlab/make_reference_plane.m`) and reproducing it in Python.

---
## 1. Entry point: `Demo_ScanAndProcess3D.m`

With `skipScanning = true` the demo only processes:

```matlab
dispersionQuadraticTerm = 8.962e+07;          % see §6: the existing TIFF used 8.949e7
[focusPositionInImageZpix, tissueRI, driftSlope] = yOCTMeasureFocusDrift(volumeOutputFolder, dispersionQuadraticTerm, 'v', true);
yOCTProcessTiledScan(volumeOutputFolder, {outputTiffFile}, ...
    'focusPositionInImageZpix', focusPositionInImageZpix, 'focusSigma', 10, ...
    'cropZRange_mm', [min(zToScan_mm) max(zToScan_mm)], ...   % [-0.03 0.04]
    'dispersionQuadraticTerm', dispersionQuadraticTerm, ...
    'outputFilePixelSize_um', 2, 'interpMethod', 'sinc5', 'v', true);
```

Note the demo's scan-geometry constants (`xOverall_mm=[-4.9 4.9]`, FOV 0.7 …) were edited
after this scan; **`ScanInfo.json` is the authoritative record** of how 10um_FOV_1 was acquired.

## 2. Call graph

```
Demo_ScanAndProcess3D
├── yOCTMeasureFocusDrift            (interactive GUI; user clicks focus band)
│   ├── yOCTProcessTiledScan_createDimStructure
│   ├── reconstructCenterBScan → yOCTLoadInterfFromFile → yOCTInterfToScanCpx → yOCTOpticalPathCorrection
│   ├── yOCTMeasureFocusDrift_fitDrift   (Theil-Sen drift fit; 1 click ⇒ constant focus)
│   └── save zChosenFocusPositions.mat / .png
└── yOCTProcessTiledScan
    ├── awsReadJSON(ScanInfo.json), yOCTUnzipTiledScan (no-op for Gan632)
    ├── yOCTProcessTiledScan_createDimStructure
    │   ├── yOCTLoadInterfFromFile(peakOnly) → _ThorlabsHeader → _ThorlabsHeaderLambda → _ThorlabsLoadChirp
    │   └── yOCTInterfToScanCpx(peakOnly) → yOCTInterfToScanCpx_getZ
    ├── parfor yI = 1:4500  (each OUTPUT y plane)
    │   ├── yOCTProcessTiledScan_getScansFromYFrame   (re-runs createDimStructure every plane!)
    │   └── for xxI = 1:12, for zzI = 1:8            (96 tiles)
    │       ├── yOCTLoadInterfFromFile(YFramesToProcess = yIInFile) → _ThorlabsData → _ReadFile
    │       ├── yOCTInterfToScanCpx → yOCTEquispaceInterf('sinc5')
    │       ├── abs
    │       ├── yOCTOpticalPathCorrection
    │       ├── yOCTProcessTiledScan_factorZ
    │       └── interp2 (×2: data·weight and weight) onto the full 35×6000 output plane
    │   └── weighted mean → mag2db → yOCT2Tif(partialFileMode=2) (one tif+json per plane)
    └── yOCT2Tif(partialFileMode=3): global clim, re-quantise every plane, write BigTIFF
```

## 3. Per-script reference

| File | Role | Key details |
|---|---|---|
| `Processing/yOCTProcessTiledScan.m` | Driver/stitcher | Defaults: `dispersionQuadraticTerm=79430000`, `focusSigma=20`, `outputFilePixelSize_um=1`, `applyPathLengthCorrection=true`. Unmatched name/values (e.g. `interpMethod`) are forwarded to loader/FFT. Asserts output pixel size == scan pixel size. `parfor` over output y planes. |
| `Processing/yOCTProcessTiledScan_createDimStructure.m` | Geometry | Tile x/y = `offset + tileRange*linspace(-.5,.5,n+1)` (last dropped). Galvo x-shift only if `galvoPhaseDelayXOffsetCorrection_mm` is **absent** (it is present ⇒ no shift). Tile z shifted so z=0 at focus pixel. Output grid spans all tile centres; output z re-zeroed at the tissue interface. |
| `Processing/yOCTProcessTiledScan_getScansFromYFrame.m` | y plane → tiles | Tiles with `gridYcc == yCenters(row)` in `octFolders` order (x-major, z-minor). Local frame = `yI - 500*(row-1)`. |
| `Processing/yOCTProcessTiledScan_factorZ.m` | Focus weight | `exp(-(z-f)^2/(2σ)^2) + exp(-4.5)` — note **(2σ)²**, i.e. effective Gaussian σ = √2·focusSigma, plus a floor. |
| `LoadSave/yOCTLoadInterfFromFile.m` | Loader front-end | Chooses Thorlabs path; subsets Y frames; **apodization subtraction**: mean of the 25 apodization A-lines subtracted from every A-line. |
| `LoadSave/yOCTLoadInterfFromFile_ThorlabsHeader.m` | Header.xml | Reads sizes, `interfSize=525`, `apodSize=25`, averaging (all 1 here). Uses `xml2struct` via `imageDatastore`. |
| `LoadSave/yOCTLoadInterfFromFile_ThorlabsHeaderLambda.m` | λ axis | `λ = 1/(chirp·(1/λmax−1/λmin)/(N−1) + 1/λmin)`; Gan632 uses λ∈[796,1010] nm (marked TODO/unmeasured in code). |
| `LoadSave/yOCTLoadInterfFromFile_ThorlabsLoadChirp.m` | Chirp | Finds `Chirp.data` in the tile folder (tries text, falls back to float32 binary). |
| `LoadSave/yOCTLoadInterfFromFile_ThorlabsData.m` | Raw read | Reads `Spectral{y-1}.data` as int16 (`'short'`), reshape `[2048, 525]`, splits apod / scan. Creates a new `imageDatastore` **per file**. |
| `LoadSave/yOCTLoadInterfFromFile_ReadFile.m` | Safe read | Missing/corrupt file ⇒ NaN data + warning. |
| `Processing/yOCTEquispaceInterf.m` | k-linearisation | `kLin = linspace(max k, min k, N)`; `sincN`: for each of the 2048 output samples a MATLAB loop sums `sinc(nq−n)` over samples with `|nq−n|<N` after removing the per-A-line mean (re-added after). λ of the new samples via `pchip`. |
| `Processing/yOCTInterfToScanCpx.m` | Spectrum → depth | Equispace if needed; Hann window normalised by RMS; dispersion phase `−β(k−mean k)²` (k in 1/nm, β = dispersionQuadraticTerm [nm²/rad]); `ifft`, keep first N/2 = 1024 bins. Options passed with `eval` (!). |
| `Processing/yOCTInterfToScanCpx_getZ.m` | Depth axis | `dz = ½·λ0²/Δλ / n` → 1.4339 µm/px in tissue (n=1.33); `z = linspace(0, dz·N/2, N/2)`. |
| `Processing/yOCTOpticalPathCorrection.m` | Field curvature | `P(x,y)=p1x+p2y+p3x²+p4y²+p5xy` (µm, tile-local); `interp2(...,'nearest')` at `z+P`; out-of-range ⇒ 0 and marked invalid (weight 0). |
| `LoadSave/yOCT2Tif.m`, `yOCT2Tif_ConvertBitsData.m` | Output | uint16 = `round((dB−c1)/(c2−c1)·65534)+1`, NaN→0; BigTIFF, PackBits, JSON metadata in tag 305 (`Software`). Mode 2 writes each plane with its own clim; mode 3 re-quantises all planes to the global clim (two quantisation steps). |
| `LoadSave/yOCTFromTif.m` | Reader | Reads JSON from **first page's** `Software` tag. |
| `Utils/yOCTChangeDimensionsStructureUnits.m` | Units | mm / µm / nm conversions of the `dim` struct. |
| `yOCTMeasureFocusDrift.m` + `_fitDrift.m` | Focus (GUI) | User clicks focus on sampled depths; Theil-Sen fit of drift vs depth → per-depth focus pixel + tissue RI estimate. For 10um_FOV_1 only one depth (z=0) qualified ⇒ constant **433 px**. |
| `Processing/yOCTFindFocusTilledScan.m` | Focus (auto) | Automatic alternative (not used for this dataset). |

Not on the tiled-reconstruction path: `yOCTProcessScan.m` (single-volume), `yOCTReslice*`, `yOCTMinMaxProjection`, `yOCTFindTissueSurface`, `yOCTEstimateScatteringCoefMuS`, `Simulation/`, `AWSUtiles/` (only path/JSON helpers are used), `ThorlabsImager/` (acquisition).

## 4. The math, step by step (per raw B-scan)

Symbols: N=2048 spectral px, nX=500 A-lines, apod=25, nZ=1024 depth px.

1. **Read**: `raw ∈ int16[525, 2048]` (A-line major).
2. **Apodization subtraction**: `I = raw[25:] − mean(raw[:25], A-lines)` → `[500, 2048]`.
3. **k-linearisation (sinc5)**: `k = 2π/λ`, `n_j = (k_j−min k)/(max k−min k)·(N−1)`, `nq_i = N−1−i`,
   `I_eq[i] = Σ_{|nq_i−n_j|<5} sinc(nq_i−n_j)·(I_j − Ī) + Ī`, Ī = per-A-line spectral mean.
4. **Window + dispersion**: `W(k) = hann(N)/rms(hann) · exp(−iβ(k_eq − mean k_eq)²)`.
5. **FFT**: `S = ifft(I_eq · W)[:1024]` (MATLAB `ifft` includes 1/N); `A = |S|`.
6. **Optical path correction**: `A_c[z,x] = A[nearest(z + P(x,y)), x]`, invalid outside the depth range.
7. **Focus weight**: `F[z,x] = (exp(−(z−f)²/(2σ)²) + e^{−4.5}) · valid[z,x]`, f=433, σ=10.
8. **Placement**: tile x = tile_x + xCenter; tile z = tile_z + zDepth − tile_z[f].
9. **Stitch** (all 96 tiles covering the plane): `num += interp2(A_c·F)`, `den += interp2(F)` (linear, 0 outside).
10. **Normalise**: `den < e^{−4.5} → NaN`; `dB = 20·log10(num/den)`.
11. **Write**: uint16 with global clim (see §3 `yOCT2Tif`).

## 5. Reconstruction parameters for 10um_FOV_1

| Parameter | Value | Source |
|---|---|---|
| octSystem | gan632 | ScanInfo.json |
| λ range | 796 – 1010 nm (from chirp) | ThorlabsHeaderLambda (Gan632 branch) |
| interpMethod | sinc5 | demo |
| dispersionQuadraticTerm | **8.949e7** (reproduces existing TIFF bit-exactly; demo says 8.962e7, ini 8.6883e7, function default 7.943e7) | this analysis, §6 |
| tissue n | 1.33 | ScanInfo.tissueRefractiveIndex |
| tile z pixel | 1.43385 µm (1024 px, 1.468 mm) | getZ |
| focusPositionInImageZpix | 433 for all 8 depths | zChosenFocusPositions.mat |
| focusSigma | 10 px (effective Gaussian σ = 14.1 px ≈ 20 µm) | demo |
| cropZRange_mm | [-0.03, 0.04] | demo (`[min max](zToScan_mm)`) |
| outputFilePixelSize_um | 2 (isotropic) | demo |
| Optical path polynomial | [0.0099168, 0.010817, −0.0001907, −0.00026705, 4.8047e−12] | ScanInfo.octProbe |
| galvo x correction | already applied at acquisition (0.01915 mm) | ScanInfo |
| Output | 4500 (y) × 35 (z) × 6000 (x) uint16, 2 µm; z ∈ [−29.5, 38.5] µm; clim [−32.87, 33.97] dB | legacy TIFF |

## 6. Validation of this understanding

* `validation/matlab/make_reference_plane.m` runs the **unmodified** legacy functions for one plane and dumps every intermediate.
* `validation/validate_stages.py`: Python vs MATLAB per stage — sinc5 2e-12, complex scan 9e-13, |scan| 2e-13 (float64); optical-path valid map 0/512 000 mismatches.
* Full plane (all 96 tiles): max |Δ| 1e-6 dB (float64), 9e-5 dB (float32) — the uint16 step is 1e-3 dB.
* Against the existing legacy output `10um_FOV_1/10um_H&E_1mmFOV.tiff`: a dispersion sweep shows a sharp optimum at **8.949e7**, where the port (with legacy double quantisation) is **bit-identical** on 99.9–100 % of pixels, max 1 LSB (rounding ties) — planes 700 and 2249. So the existing TIFF was made with 8.949e7, not the 8.962e7 in the demo script.

## 7. Where the time goes (legacy, measured)

One output plane (96 B-scans), MATLAB R2024a, 8 threads, `make_reference_plane.m`:

| Stage | Time | Share |
|---|---|---|
| `yOCTInterfToScanCpx` (sinc5 loop + FFT) | 15.74 s | 77 % |
| `yOCTLoadInterfFromFile` (96 files, datastore per file) | 1.56 s | 8 % |
| `createDimStructure` + `getScansFromYFrame` (repeated per plane) | 2.43 s* | — |
| `yOCTOpticalPathCorrection` | 1.04 s | 5 % |
| 2× `interp2` onto full 35×6000 plane ×96 | 0.71 s | 3.5 % |
| **Plane total** | **20.4 s** | |

\* `getScansFromYFrame` (1.0 s) runs every plane; createDimStructure once more inside it.

4500 planes × 20.4 s ≈ **25.5 h serial**; the `parfor` over 8 workers helps but each worker is itself compute-bound on the same 8 cores, so realistic wall-clock is several hours. See `docs/03_acceleration.md`.

## 8. Known quirks / latent bugs in the legacy code (kept for fidelity unless noted)

* `factorZ` uses `(2σ)²` in the denominator (effective σ is √2 larger than the name implies).
* `yOCTOpticalPathCorrection` overwrites `correctedScanValidDataMap` for each y (harmless: called with one frame).
* `yOCTInterfToScanCpx` parses options with `eval`.
* Header, JSON and chirp are re-read for every output plane (in `getScansFromYFrame`).
* Double uint16 quantisation (per-plane clim, then global clim) adds up to ~1 extra LSB error.
* `clim` uses `min/max` over all values; a pixel with `num=0` would give −Inf and break scaling (not observed).
* Gan632 λ range is a placeholder copied from Ganymede ("TODO … measured"): affects the absolute z scale (1.4339 µm/px) of every reconstruction, not just the port.
