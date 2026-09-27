"""Preview / diagnostic helpers for the web UI (single B-scans and estimator plots).

preview_bscan   one tile B-scan in dB, processed like the reconstruction
                (apod, sinc5, dispersion, |ifft|, optional optical-path correction) -
                equivalent to reconstructCenterBScan in yOCTMeasureFocusDrift.m.
render_png      grayscale PNG of a dB image with optional focus line(s).
dispersion_curve_png / focus_profile_png   diagnostic plots of the estimator results.
"""
from __future__ import annotations

import io

import numpy as np

from ..params import FOCUS_FILE, load_focus_file, resolve_dispersion
from ._common import VolumeContext, to_py

MAX_COLS = 500
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
S1, S2, S3, S4 = "#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7"


def preview_bscan(volume_folder, zi, xi=None, yi=None, frame=None, dispersion_quadratic_term=None,
                  device="cpu", apply_opc=True) -> dict:
    """-> dict(image_db (nZ, nX<=500) float32, z_px_count, tile, frame, xi, yi, zi, z_depth_mm,
    dispersion_quadratic_term, z_um (axis), x_mm (axis), focus_hint (1-based pixel or None))."""
    from .focus import apply_opc as _opc
    with VolumeContext(volume_folder, device=device) as ctx:
        si = ctx.si
        zi = int(zi)
        if not 0 <= zi < len(si.z_depths_mm):
            raise ValueError(f"zi must be in [0, {len(si.z_depths_mm) - 1}]")
        xi0, yi0 = ctx.central_indices()
        xi = xi0 if xi is None else int(xi)
        yi = yi0 if yi is None else int(yi)
        t = ctx.by_index.get((xi, yi, zi))
        if t is None:
            raise ValueError(f"no tile at (xi={xi}, yi={yi}, zi={zi})")
        frame = (ctx.n_frames - 1) // 2 if frame is None else int(frame)
        if dispersion_quadratic_term in (None, "", "auto"):
            disp = resolve_dispersion(si)[0]
        else:
            disp = float(dispersion_quadratic_term)
        mag = ctx.magnitude(t.folder, [frame], disp)
        mag = (mag.get() if hasattr(mag, "get") else np.asarray(mag)).astype(float)
        if apply_opc:
            mag = _opc(ctx, mag, [frame])
        img = mag[0].T                                             # (nZ, nX)
        nx = img.shape[1]
        step = int(np.ceil(nx / MAX_COLS))
        if step > 1:                                               # block-average x
            n = nx // step * step
            img = np.nanmean(img[:, :n].reshape(img.shape[0], -1, step), axis=2)
        with np.errstate(divide="ignore", invalid="ignore"):
            db = (20 * np.log10(img)).astype(np.float32)
        sg = ctx.geometry(disp)
        hint = None
        fp = ctx.volume / FOCUS_FILE
        if fp.exists():
            try:
                f = load_focus_file(fp)
                hint = float(f[zi] if f.size > zi else f[0])
            except Exception:
                hint = None
        x_mm = t.x_center_mm + si.tile_range_x_mm * (np.arange(0, nx, step)[: db.shape[1]] / nx - 0.5)
        return {"image_db": db, "z_px_count": int(db.shape[0]), "tile": t.folder, "frame": frame,
                "xi": xi, "yi": yi, "zi": zi, "z_depth_mm": float(t.z_depth_mm),
                "dispersion_quadratic_term": disp, "optical_path_corrected": bool(apply_opc),
                "z_um": to_py(sg.z_um), "x_mm": to_py(x_mm), "focus_hint": hint}


def _fig_bytes(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=fig.dpi, facecolor=fig.get_facecolor())
    import matplotlib.pyplot as plt
    plt.close(fig)
    return buf.getvalue()


def _style(ax):
    ax.set_facecolor(SURF)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)


def render_png(image_db, focus_pix=None, clim=None, width_px=700) -> bytes:
    """Grayscale PNG of a dB image (rows = depth pixels). focus_pix: 1-based pixel or list."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    img = np.asarray(image_db, float)
    fin = img[np.isfinite(img)]
    if clim is None:
        clim = (np.percentile(fin, 5), np.percentile(fin, 99.8)) if fin.size else (0, 1)
    nz, nx = img.shape
    h_px = int(width_px * min(2.0, max(0.5, nz / max(nx, 1) * 0.5)))
    fig = plt.figure(figsize=(width_px / 100, h_px / 100), dpi=100, facecolor=SURF)
    ax = fig.add_axes([0.08, 0.06, 0.9, 0.9])
    ax.imshow(np.nan_to_num(img, nan=clim[0]), cmap="gray", vmin=clim[0], vmax=clim[1],
              aspect="auto", interpolation="nearest")
    ax.set_xlabel("x [px]", fontsize=8, color=INK2)
    ax.set_ylabel("z [px]", fontsize=8, color=INK2)
    ax.tick_params(colors=INK2, labelsize=8)
    if focus_pix is not None:
        fps = np.atleast_1d(np.asarray(focus_pix, float))
        for f in np.unique(fps[np.isfinite(fps)]):
            ax.axhline(f - 1, color="#eda100", lw=1.2, ls="--")
            ax.text(2, f - 1 + 6, f"focus {f:g}", color="#eda100", fontsize=8, ha="left", va="top",
                    bbox=dict(facecolor="black", alpha=0.6, edgecolor="none", pad=1.5))
    return _fig_bytes(fig)


def dispersion_curve_png(result) -> bytes:
    """Normalised sharpness vs dispersion: per-B-scan curves (thin), mean (bold), estimate."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    b = np.asarray(result["candidates"], float) / 1e7
    fig, ax = plt.subplots(figsize=(7, 3.6), dpi=100, facecolor=SURF)
    _style(ax)
    for c in result.get("score_curves_per_bscan", []):
        ax.plot(b, c, color="#86b6ef", lw=0.8, alpha=0.7)
    ax.plot(b, result["score_curve"], color=S1, lw=2, label="mean of normalised B-scan scores")
    v = result["value"] / 1e7
    ax.axvline(v, color=INK, lw=1.2, label=f"estimate {result['value']:.4g}")
    if result.get("initial") is not None:
        ax.axvline(result["initial"] / 1e7, color=INK2, lw=1, ls=":", label=f"initial {result['initial']:.4g}")
    per = np.asarray(result.get("per_bscan_values", []), float) / 1e7
    if per.size:
        ax.plot(per, np.full(per.size, -0.05), "|", color=S2, ms=10, mew=1.5, label="per-B-scan optima")
    ax.set_xlabel("dispersionQuadraticTerm [1e7 nm²/rad]", fontsize=9, color=INK2)
    ax.set_ylabel("normalised score", fontsize=9, color=INK2)
    ax.set_title(f"Dispersion search ({result.get('metric', '')})", fontsize=10, color=INK, loc="left")
    ax.set_ylim(-0.1, 1.05)
    ax.legend(fontsize=7, frameon=False, loc="lower left")
    fig.tight_layout()
    return _fig_bytes(fig)


def focus_profile_png(result) -> bytes:
    """Depth profiles used by detect_focus with the detected positions."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 3.6), dpi=100, facecolor=SURF)
    _style(ax)
    leg = result.get("legacy", {})

    def db(v):
        v = np.asarray(v, float)
        with np.errstate(divide="ignore", invalid="ignore"):
            return 20 * np.log10(v)
    z = None
    if leg.get("step1", {}).get("profile"):
        p = db(leg["step1"]["profile"]); z = np.arange(1, len(p) + 1)
        ax.plot(z, p, color=S2, lw=1, label=f"interface tile z={leg['step1']['z_depth_mm'] * 1e3:.0f} um")
    if leg.get("step2", {}).get("profile_smoothed"):
        p = db(leg["step2"]["profile_smoothed"]); z = np.arange(1, len(p) + 1)
        ax.plot(z, p, color=S3, lw=1, label=f"gel tile z={leg['step2']['z_depth_mm'] * 1e3:.0f} um (smoothed)")
    for m in (result.get("drift") or {}).get("measurements", [])[:6]:
        p = db(m["profile"]); z = np.arange(1, len(p) + 1)
        ax.plot(z, p, color=S1, lw=1, alpha=0.8, label=f"measured z={m['z_depth_mm'] * 1e3:.0f} um (OPC)")
    fps = sorted(set(result.get("focus_positions", [])))
    for f in fps:
        ax.axvline(f, color=INK, lw=1.2)
    if fps:
        ax.text(fps[0], ax.get_ylim()[1], f" focus {fps[0]}" + (f"..{fps[-1]}" if len(fps) > 1 else ""),
                fontsize=8, color=INK, va="top")
    if "step2" in result:
        ax.axvline(result["step2"], color=S3, lw=1, ls="--", label=f"legacy step 2: {result['step2']}")
    cp = result.get("confocal_peak") or {}
    if "value" in cp:
        ax.axvline(cp["value"], color=S4, lw=1, ls=":", label=f"stationary confocal peak: {cp['value']}")
    if z is not None:
        c = fps[0] if fps else len(z) // 2
        ax.set_xlim(max(1, c - 200), min(len(z), c + 200))
    ax.set_xlabel("z [px, 1-based]", fontsize=9, color=INK2)
    ax.set_ylabel("median-x |A| [dB]", fontsize=9, color=INK2)
    ax.set_title("Focus detection", fontsize=10, color=INK, loc="left")
    ax.legend(fontsize=7, frameon=False, loc="upper right")
    fig.tight_layout()
    return _fig_bytes(fig)
