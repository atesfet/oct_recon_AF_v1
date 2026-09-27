# 04 — Reconstruction inputs & parameters (what is needed, where it comes from, what can be estimated)

Applies to `octrecon` (oct_recon_AF_v1) and the legacy `yOCTProcessTiledScan.m` — they take the same inputs.

## 1. Required inputs at a glance

| # | Input | Source | Stored with the raw data? | Value for 10um_FOV_1 |
|---|---|---|---|---|
| 1 | Raw tiles `DataNN/data/Spectral*.data` | acquisition | ✅ | 864 tiles |
| 2 | Tile header (`Header.xml`): spectral px, A-lines, apodization lines, B-scans, averaging | acquisition | ✅ | 2048 / 525 / 25 / 500 / 1 |
| 3 | Chirp (`data/Chirp.data`) | acquisition (system calibration) | ✅ | float32[2048] |
| 4 | Scan geometry (`ScanInfo.json`): tile grid, centres, z-depths, pixel size, tile size | acquisition | ✅ | 12×9×8, 2 µm, 1 mm |
| 5 | Tissue refractive index `n` | `ScanInfo.tissueRefractiveIndex` (typed at scan time) | ✅ | 1.33 |
| 6 | Optical path (field curvature) polynomial | `ScanInfo.octProbe` (probe .ini) | ✅ | 5 coefficients |
| 7 | **Focus pixel per depth** `focusPositionInImageZpix` | `zChosenFocusPositions.mat` from **interactive** `yOCTMeasureFocusDrift` | ⚠️ in the volume folder, but *derived* (needs a human click today) | 433 (all depths) |
| 8 | **Dispersion** `dispersionQuadraticTerm` β [nm²/rad] | user/calibration | ❌ (only a probe default 8.6883e7 is in ScanInfo) | **8.949e7** |
| 9 | `focusSigma` (z-stitch blend width, px) | user, per objective | ❌ | 10 |
| 10 | `interpMethod` (k-linearisation) | user | ❌ | `sinc5` |
| 11 | `cropZRange_mm` | user | ❌ (but = [min, max] of `zDepths` in the demo) | [-0.03, 0.04] |
| 12 | `outputFilePixelSize_um` | user | ❌ (must equal `pixelSize_um`, legacy restriction) | 2 |
| 13 | λ range of the spectrometer | **hard-coded** per system in `yOCTLoadInterfFromFile_ThorlabsHeaderLambda.m` | ❌ | 796–1010 nm (gan632, marked TODO) |
| — | `applyPathLengthCorrection` | user | — | true |

In `octrecon` items 7–12 live in `configs/<dataset>.yaml`; items 1–6 are read automatically; item 13 is in `octrecon/core/geometry.py:LAMBDA_RANGE_NM`.

**For a new scan with the same system + probe + objective**, only the paths and the focus file change;
β, σ, interpolation and λ range are per-system/probe constants.

## 2. Parameter details and sensitivity

### Dispersion β (`dispersion_quadratic_term`) — the critical one
* Enters as phase `−β(k−k̄)²` (k in 1/nm) before the FFT; wrong β blurs every A-scan axially.
* Candidates found in the code: 7.943e7 (function default), 8.6883e7 (probe .ini → ScanInfo), 8.962e7 (Demo script), and **8.949e7** — the value that reproduces the existing legacy TIFF bit-for-bit.
* Sensitivity (mean |ΔdB| vs legacy TIFF): 7.943e7 → 1.62 dB, 8.6883e7 → 0.48 dB, 8.962e7 → 0.024 dB, 9.2e7 → 0.46 dB, 8.949e7 → exact. A 0.1 % change is visible at the LSB level; 3 % is visibly blurrier.
* Sign depends on the scan head (OCTP900 probes are negative, OCTG positive).

### Focus pixel (`focus_positions`)
* Sets tile z=0, the centre of the Gaussian stitching weight, and re-centres each depth on its own focus.
* For this dataset: a single click at z=0 (scan shallower than `firstMeasureDepth_um=50` ⇒ only one depth offered) ⇒ constant 433, no drift.
* Accepted as the `.mat` path, a scalar, or a per-depth list.

### `focus_sigma`
* Weight = `exp(−(z−f)²/(2σ)²) + e^{−4.5}` ⇒ the effective Gaussian std is **√2·σ** (≈14 px ≈ 20 µm for σ=10).
* Code comments: 10x→20, 20x→10, 40x→5–10. With a 10 µm z-step (≈7 px), σ=10 gives heavy overlap between depths.

### z scale caveat (λ range)
* The z pixel (1.4339 µm in tissue) follows from λ ∈ [796, 1010] nm, which is copied from a **Ganymede** calibration (`Reports/2019-08-31GanymedeFilterCalibration.pdf`) and marked "TODO … measured" for gan632.
* `Header.xml` of this dataset reports CentralWavelength 880 nm / Bandwidth 239.46 nm (Thorlabs' own `RangeZ` 1.6558 mm matches that exactly) ⇒ **~15 % different z pixel size**.
* This affects legacy and port equally (the port reproduces legacy). Changing it would also change the optimal β. Resolving it needs a calibration (glass-slide stage sweep, see §3a, or a known-thickness target).

## 3. What in the codebase can estimate these parameters

| Parameter | Existing code | Offline on an existing scan? | Automatic? | Port effort / recommendation |
|---|---|---|---|---|
| β (dispersion) | `Applications/yOCTScanGlassSlideToFindFocusAndDispersionQuadraticTerm.m` (l.146-155): `fminsearch` over β maximising the peak of the x-averaged log A-scan | Core math yes; script itself scans a glass slide (hardware) | Yes | **Best candidate.** ~30 lines; run a bounded 1-D search (e.g. 5e7–1.3e8 for OCTG) on ~8 B-scans at the z=0 interface + tissue tiles, sharpness metric (peak height / entropy), take the median. One FFT per B-scan per candidate ⇒ seconds on GPU. Validate: should recover ≈8.949e7. |
| β (manual) | `Demo_DispersionCorrectionManual.m`: slider, judged by eye | Yes | No (user) | Replace with the automatic search. |
| focus pixel (single) | `Processing/yOCTFindFocusTilledScan.m`: brightest layer near z=0 (5 frames, median over x), then refine on a gel-only depth by smoothed peak (`findpeaks`, σ=4 px); optional click | **Yes** | Yes with `manualRefinment=false` | Easy port (median + Gaussian + `find_peaks`); use all gel depths (−30,−20,−10 µm here) and several tiles. Validate vs 433. |
| focus per depth + tissue RI | `yOCTMeasureFocusDrift.m` (GUI clicks) + `yOCTMeasureFocusDrift_fitDrift.m` (Theil–Sen fit with MAD outlier rejection, physical slope bounds, `n_s = sqrt(n_i² + m·dz·n_i·n_a)`) | Yes | Fit yes; clicks no | Port the fit (~80 lines) and feed it automatic per-depth focus detections. Only meaningful for deep scans in index-mismatched tissue (not this 40 µm scan). **Only in-code estimator of tissue RI.** |
| z-scale / n·λ0²/Δλ | Glass-slide script step 5 (stage motion vs interface pixel fit) | Needs a glass-slide z sweep | Yes | Use once per system to calibrate the λ range / z pixel. |
| cropZRange | `Processing/yOCTFindTissueSurface.m` (Otsu threshold per y, first sustained crossing, smoothing) | Yes, after a reconstruction | Yes | Could set crop = surface − margin … surface + depth. Today the default rule `[min, max](zDepths)` is sufficient. |
| focus QA | `Processing/yOCTAssertFocusAndComputeZOffset.m`, `Demo_CheckFocusInTileScan.m` | Yes | QA only | Useful as a post-check, not an estimator. |
| focusSigma | none (hard-coded per objective) | — | — | Could be fitted from the axial width of the focus band across the z-stack. |
| interpMethod, output pixel size | none (user choices) | — | — | Keep `sinc5`; pixel size = `ScanInfo.pixelSize_um`. |

Latent bug noted: `Applications/yOCTTissueSurfaceAutofocus.m:155` unpacks `yOCTAssertFocusAndComputeZOffset` outputs in the wrong order.

## 4. Minimal-input recipe (proposed next step)

1. `crop_z_range_mm` ← `[min(zDepths), max(zDepths)]`, `output_pixel_size_um` ← `ScanInfo.pixelSize_um` (derivable today).
2. `focus_positions` ← automatic gel-band detector (port of `yOCTFindFocusTilledScan`), keep the `.mat` as override.
3. `dispersion_quadratic_term` ← automatic sharpness search seeded from `octProbe.DefaultDispersionQuadraticTerm`, stored per probe once validated.
4. Keep `focus_sigma` and `interp_method` as per-objective constants in the config.

With 1–3 the reconstruction needs **only the raw volume folder + a per-probe config**.
