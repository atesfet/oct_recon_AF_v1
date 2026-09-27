"""Compare a full reconstruction TIFF with the legacy MATLAB TIFF, plane by plane.

  python validation/validate_full_volume.py \
      --new outputs/10um_FOV_1/10um_FOV_1_recon.tiff \
      --legacy "10um_FOV_1/10um_H&E_1mmFOV.tiff" \
      --out benchmarks/results/full_volume_validation.json
Also writes example comparison images (B-scan + en-face) next to the JSON.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import tifffile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from octrecon.io.tiff_writer import bits_to_db  # noqa: E402


def clim_of(t):
    return json.loads(t.pages[0].tags[305].value)["clim"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--new", default=str(ROOT / "outputs/10um_FOV_1/10um_FOV_1_recon.tiff"))
    ap.add_argument("--legacy", default=str(ROOT / "10um_FOV_1/10um_H&E_1mmFOV.tiff"))
    ap.add_argument("--out", default=str(ROOT / "benchmarks/results/full_volume_validation.json"))
    a = ap.parse_args()
    out = Path(a.out)
    with tifffile.TiffFile(a.new) as tn, tifffile.TiffFile(a.legacy) as tl:
        cn, cl = clim_of(tn), clim_of(tl)
        ny = len(tn.pages)
        assert ny == len(tl.pages), (ny, len(tl.pages))
        step = (cl[1] - cl[0]) / 65534
        hist = np.zeros(200)  # |ΔdB| histogram in LSB units
        nan_mismatch = 0
        n_px = 0
        max_abs = 0.0
        sum_abs = 0.0
        per_plane_max = np.zeros(ny)
        enface_new = enface_old = None
        for y in range(ny):
            dn = bits_to_db(tn.pages[y].asarray(), cn)
            dl = bits_to_db(tl.pages[y].asarray(), cl)
            if y == 0:
                enface_new = np.empty((ny, dn.shape[1]), np.float32)
                enface_old = np.empty_like(enface_new)
                zc = int(np.argmin(np.abs(np.array(json.loads(tn.pages[0].tags[305].value)["metadata"]["z"]["values"]))))
            enface_new[y] = dn[zc]
            enface_old[y] = dl[zc]
            nan_mismatch += int((np.isnan(dn) != np.isnan(dl)).sum())
            d = np.abs(dn - dl)
            d = d[np.isfinite(d)]
            n_px += d.size
            if d.size:
                m = float(d.max())
                per_plane_max[y] = m
                max_abs = max(max_abs, m)
                sum_abs += float(d.sum())
                hist += np.histogram(np.minimum(d / step, 199.5), bins=200, range=(0, 200))[0]
            if y == 2249:
                bscan_new, bscan_old = dn, dl
        res = {
            "new": a.new, "legacy": a.legacy, "planes": ny, "plane_shape_zx": list(dn.shape),
            "clim_new_dB": cn, "clim_legacy_dB": cl, "lsb_dB": step,
            "nan_mask_mismatches": nan_mismatch, "compared_pixels": n_px,
            "max_abs_diff_dB": max_abs, "mean_abs_diff_dB": sum_abs / max(n_px, 1),
            "frac_within_0.5_lsb": float(hist[:1].sum() / n_px),
            "frac_within_1_lsb": float(hist[:2].sum() / n_px),
            "frac_within_2_lsb": float(hist[:3].sum() / n_px),
            "worst_planes": [int(i) for i in np.argsort(per_plane_max)[-5:][::-1]],
            "worst_plane_max_dB": float(per_plane_max.max()),
        }
    print(json.dumps(res, indent=2))
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(res, f, indent=2)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    lo, hi = np.nanpercentile(enface_old, [1, 99.5])
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    for k, (im, t) in enumerate([(enface_old, "legacy MATLAB"), (enface_new, "oct_recon_AF_v1 (GPU)")]):
        ax[k].imshow(im, cmap="gray", vmin=lo, vmax=hi, extent=[-6, 6, 4.5, -4.5])
        ax[k].set_title(f"en-face z≈0 µm — {t}")
        ax[k].set_xlabel("x [mm]"); ax[k].set_ylabel("y [mm]")
    dd = ax[2].imshow(enface_new - enface_old, cmap="coolwarm", vmin=-5 * step, vmax=5 * step, extent=[-6, 6, 4.5, -4.5])
    ax[2].set_title("difference [dB] (±5 LSB)")
    fig.colorbar(dd, ax=ax[2])
    fig.tight_layout(); fig.savefig(out.with_name("full_volume_enface_comparison.png"), dpi=130)
    fig, ax = plt.subplots(3, 1, figsize=(14, 5.5))
    for k, (im, t) in enumerate([(bscan_old, "legacy"), (bscan_new, "GPU port")]):
        ax[k].imshow(im, cmap="gray", vmin=lo, vmax=hi, aspect="auto"); ax[k].set_title(f"B-scan y=2250 — {t}")
    ax[2].imshow(bscan_new - bscan_old, cmap="coolwarm", vmin=-5 * step, vmax=5 * step, aspect="auto")
    ax[2].set_title("difference (±5 LSB)")
    fig.tight_layout(); fig.savefig(out.with_name("full_volume_bscan_comparison.png"), dpi=130)


if __name__ == "__main__":
    main()
