"""Automatic (offline) estimation of the quadratic dispersion term (dispersionQuadraticTerm).

Legacy reference
----------------
* Applications/yOCTScanGlassSlideToFindFocusAndDispersionQuadraticTerm.m:146-155
  (hardware procedure: scan a glass slide, then
  ``fminsearch(@(d) -max(mean(log|scan(d)|, x)), initialGuess)``).  That metric is
  implemented here as ``metric='legacy_peak'``; it assumes one dominant specular
  reflector and does NOT work on tissue B-scans (see docs/05_parameter_estimation.md).
* Demo_DispersionCorrectionManual.m: manual slider (visual sharpness).

Algorithm (no hardware, works on an existing tiled scan)
--------------------------------------------------------
1. Pick B-scans automatically: tiles inside the tissue (zDepth > 0, else >= 0) in the
   central part of the xy grid, frames spread over the tile; 2x more candidates are
   read and the ones with the most signal (bright-pixel dynamic range) are kept.
2. k-linearise each B-scan ONCE (apodization, sinc5 - same code as the reconstruction)
   and multiply by the Hann/rms window.  Only the dispersion phase depends on beta:
   ``scan(beta) = ifft(X * exp(-i*beta*(k-k0)^2))`` -> every candidate costs one FFT.
3. Score each B-scan with an image-sharpness metric, normalise each B-scan's curve to
   [0, 1] over a coarse grid, average the normalised curves, and refine the maximum of
   the average with a bounded scalar search (scipy ``minimize_scalar``, method='bounded').
   Per-B-scan optima are refined the same way and reported for spread / stability.

Metrics (all maximised; the first ~20 depth pixels (DC / autocorrelation) are ignored)
  top_percentile  mean of the brightest 1 % of intensity pixels |A|^2 (default, 'auto')
  kurtosis        sum(I^2) / sum(I)^2       (normalised 4th moment of |A|)
  entropy         -(-sum p log p), p = I / sum(I)
  legacy_peak     max_z mean_x log|A|        (legacy glass-slide metric)
"""
from __future__ import annotations

import time

import numpy as np

from ..params import resolve_dispersion
from ._common import VolumeContext, to_py

Z_MIN_PIX = 20              # ignore DC / autocorrelation rows
TOP_FRACTION = 0.01         # top_percentile metric
METRICS = ("top_percentile", "kurtosis", "entropy", "legacy_peak")


# --------------------------------------------------------------------------- metrics
def _metric(xp, A, name):
    """A: (B, nX, nZ) magnitude -> (B,) score (higher = better focus/compensation)."""
    B = A.shape[0]
    if name == "legacy_peak":
        L = xp.log(A[:, :, Z_MIN_PIX:] + 1e-12).mean(axis=1)          # mean over x
        return L.max(axis=1)
    I = (A[:, :, Z_MIN_PIX:] ** 2).reshape(B, -1).astype(xp.float64)
    if name == "top_percentile":
        M = I.shape[1]
        k = max(1, int(M * TOP_FRACTION))
        top = xp.partition(I, M - k, axis=1)[:, M - k:]
        return xp.log(top.mean(axis=1))
    if name == "kurtosis":
        return (I ** 2).sum(axis=1) / I.sum(axis=1) ** 2
    if name == "entropy":
        p = I / I.sum(axis=1, keepdims=True)
        return (p * xp.log(p + 1e-30)).sum(axis=1)
    raise ValueError(f"unknown metric {name!r}; choose from {METRICS} or 'auto'")


class DispersionScorer:
    """Holds the k-linearised, windowed spectra of a few B-scans; scores any beta."""

    def __init__(self, ctx: VolumeContext, raw_list, metric: str):
        xp = self.xp = ctx.xp
        self.metric = metric
        sg = ctx.geometry(0.0)
        sp = ctx.processor(0.0)
        N = len(sg.lambda_nm)
        self.nz = N // 2
        hann = 0.5 * (1 - np.cos(2 * np.pi * np.arange(N) / (N - 1)))
        hann = hann / np.sqrt(np.mean(hann ** 2))
        k_eq = 2 * np.pi / sg.lambda_eq_nm                 # same k axis as build_spectral_geometry
        self.dk2 = xp.asarray((k_eq - k_eq.mean()) ** 2)   # float64
        specs = []
        self.nrep = max(1, int(ctx.hdr.bscan_avg))
        for raw in raw_list:
            eq = sp.linearise(xp.asarray(raw))             # (rep, nX, N) real
            specs.append((eq * xp.asarray(hann)).astype(xp.complex64))
        self.X = xp.stack(specs)                            # (B, rep, nX, N)
        self.n_evals = 0

    def window(self, beta):
        return self.xp.exp(-1j * float(beta) * self.dk2).astype(self.xp.complex64)

    def magnitude(self, beta, idx=None):
        xp = self.xp
        X = self.X if idx is None else self.X[idx:idx + 1]
        spec = X * self.window(beta)
        if xp is np:
            import os
            import scipy.fft
            ft = scipy.fft.ifft(spec, axis=-1, workers=os.cpu_count(), overwrite_x=True)
        else:
            ft = xp.fft.ifft(spec, axis=-1)
        A = xp.abs(ft[..., : self.nz]).mean(axis=1)         # mean |.| over B-scan repeats
        return A

    def scores(self, beta, idx=None) -> np.ndarray:
        self.n_evals += 1
        s = _metric(self.xp, self.magnitude(beta, idx), self.metric)
        return np.asarray(s.get() if hasattr(s, "get") else s, float)


# --------------------------------------------------------------------------- B-scan selection
def select_bscans(ctx: VolumeContext, n: int, rng_seed: int = 0):
    """Candidate (folder, frame) pairs: tissue depths, central tiles, spread frames."""
    si = ctx.si
    z = si.z_depths_mm
    depths = [i for i in range(len(z)) if z[i] > 1e-9]
    if not depths:
        depths = [i for i in range(len(z)) if z[i] >= -1e-9] or list(range(len(z)))

    def central(c):
        c = np.asarray(c)
        if len(c) <= 2:
            return list(range(len(c)))
        span = np.ptp(c)
        ok = np.nonzero(np.abs(c - c.mean()) <= 0.3 * span + 1e-9)[0]
        return list(ok) if len(ok) else [int(np.argmin(np.abs(c - c.mean())))]

    xs, ys = central(si.x_centers_mm), central(si.y_centers_mm)
    rng = np.random.default_rng(rng_seed)
    nf = ctx.n_frames
    frames = np.unique(np.round(np.linspace(0.1, 0.9, max(n, 2)) * (nf - 1)).astype(int))
    out, seen = [], set()
    for i in range(4 * n):
        if len(out) >= n:
            break
        zi = depths[i % len(depths)]
        t = ctx.by_index.get((int(rng.choice(xs)), int(rng.choice(ys)), zi))
        fr = int(frames[i % len(frames)]) if nf > 1 else 0
        if t is None or (t.folder, fr) in seen:
            continue
        seen.add((t.folder, fr))
        out.append((t.folder, fr))
    return out


def _signal_score(ctx, sp, raw):
    """Bright-pixel dynamic range (dB) of one B-scan: rejects empty (no tissue) tiles."""
    A = sp.magnitude(ctx.xp.asarray(raw))
    A = A.get() if hasattr(A, "get") else A
    I = A[..., Z_MIN_PIX:].ravel().astype(float) ** 2
    return 10 * np.log10(np.percentile(I, 99.5) / (np.median(I) + 1e-30) + 1e-30)


# --------------------------------------------------------------------------- main API
def estimate_dispersion(volume_folder, device="cpu", initial=None, bounds=None, n_bscans=8,
                        tiles=None, metric="auto", log=print, n_grid=41) -> dict:
    """Estimate dispersionQuadraticTerm [nm^2/rad] from an existing tiled scan.

    initial : starting value; default params.resolve_dispersion(ScanInfo) (probe preset,
              then probe .ini default, then legacy default).  Only used to derive bounds.
    bounds  : (lo, hi); default sign(initial) * |initial| * [0.5, 1.5].
    tiles   : optional explicit list of folder names or (folder, frame) pairs.
    metric  : 'auto' (= 'top_percentile') | 'top_percentile' | 'kurtosis' | 'entropy' | 'legacy_peak'.
    Returns a JSON-friendly dict (see keys below).
    """
    from scipy.optimize import minimize_scalar

    log = log or (lambda m: None)
    t_start = time.perf_counter()
    metric_used = "top_percentile" if metric in (None, "", "auto") else metric
    if metric_used not in METRICS:
        raise ValueError(f"unknown metric {metric!r}")
    ctx = VolumeContext(volume_folder, device=device)
    try:
        if initial in (None, "", "auto"):
            initial, init_src = resolve_dispersion(ctx.si)
        else:
            initial, init_src = float(initial), "user"
        if bounds is None:
            lo, hi = sorted((0.5 * initial, 1.5 * initial)) if initial != 0 else (-3e8, 3e8)
        else:
            lo, hi = sorted(float(b) for b in bounds)
        log(f"dispersion search: metric={metric_used} initial={initial:.4g} ({init_src}) "
            f"bounds=[{lo:.4g}, {hi:.4g}] device={ctx.device}")

        # ---- choose B-scans
        if tiles:
            pairs = []
            for i, t in enumerate(tiles):
                if isinstance(t, (list, tuple)):
                    pairs.append((str(t[0]), int(t[1])))
                else:
                    pairs.append((str(t), ctx.n_frames // 2))
            cands = pairs
        else:
            cands = select_bscans(ctx, 2 * n_bscans)
        sp0 = ctx.processor(initial)
        raws, sig = [], []
        for folder, fr in cands:
            raw = ctx.read_raw(folder, [fr])
            raws.append(raw)
            sig.append(_signal_score(ctx, sp0, raw))
        order = np.argsort(sig)[::-1][: n_bscans] if not tiles else np.arange(len(cands))
        order = sorted(int(i) for i in order)
        used = [{"folder": cands[i][0], "frame": cands[i][1],
                 "z_depth_mm": float(ctx.tile(cands[i][0]).z_depth_mm),
                 "signal_dB": float(sig[i])} for i in order]
        for u in used:
            log(f"  B-scan {u['folder']} frame {u['frame']} (z={u['z_depth_mm'] * 1e3:.0f} um, "
                f"signal {u['signal_dB']:.1f} dB)")
        scorer = DispersionScorer(ctx, [raws[i] for i in order], metric_used)
        B = len(order)

        # ---- coarse grid
        grid = np.linspace(lo, hi, int(n_grid))
        S = np.array([scorer.scores(b) for b in grid])        # (G, B)
        smin, smax = S.min(0), S.max(0)
        rngs = np.where(smax > smin, smax - smin, 1.0)
        Sn = (S - smin) / rngs
        agg = Sn.mean(1)
        gi = int(np.argmax(agg))

        def bracket(i):
            return grid[max(i - 1, 0)], grid[min(i + 1, len(grid) - 1)]

        tol = max(abs(initial), abs(lo), abs(hi)) * 2e-5
        a, b = bracket(gi)
        r = minimize_scalar(lambda x: -float(np.mean((scorer.scores(x) - smin) / rngs)),
                            bounds=(a, b), method="bounded", options={"xatol": tol})
        value = float(r.x) if -r.fun >= agg[gi] else float(grid[gi])

        # ---- per-B-scan optima (spread / stability)
        per = []
        for j in range(B):
            i = int(np.argmax(S[:, j]))
            a, b = bracket(i)
            rj = minimize_scalar(lambda x: -float(scorer.scores(x, j)[0]), bounds=(a, b),
                                 method="bounded", options={"xatol": tol})
            per.append(float(rj.x) if -rj.fun >= S[i, j] else float(grid[i]))
        per = np.array(per)
        at_bound = bool(value <= lo + (hi - lo) / (n_grid - 1) or value >= hi - (hi - lo) / (n_grid - 1))
        el = time.perf_counter() - t_start
        mad = float(1.4826 * np.median(np.abs(per - np.median(per))))
        log(f"dispersion estimate = {value:.5g} (median of per-B-scan optima {np.median(per):.5g}, "
            f"robust spread {mad:.3g}; {scorer.n_evals} evaluations, {el:.1f} s)")
        if at_bound:
            log("WARNING: optimum at the edge of the search bounds - widen `bounds`")
        return to_py({
            "value": value,
            "method": f"offline sharpness search ({metric_used}) on {B} B-scans; coarse grid "
                      f"{n_grid} pts + bounded refinement of the mean normalised score",
            "metric": metric_used,
            "initial": float(initial), "initial_source": init_src,
            "bounds": [float(lo), float(hi)],
            "candidates": grid.tolist(),
            "score_curve": agg.tolist(),
            "score_curves_per_bscan": Sn.T.tolist(),
            "per_bscan_values": per.tolist(),
            "median_per_bscan": float(np.median(per)),
            "spread_mad": mad,
            "at_bound": at_bound,
            "tiles_used": used,
            "n_evaluations": scorer.n_evals,
            "device": ctx.device,
            "elapsed_s": el,
        })
    finally:
        ctx.close()
