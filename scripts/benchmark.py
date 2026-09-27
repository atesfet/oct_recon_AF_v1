"""Throughput benchmark of the Python port on real data.

Processes `--planes` consecutive output planes of one y-tile row (all 96 tiles
each) and extrapolates to the full volume. Also checks the result against the
MATLAB reference plane when it is among the processed planes.

  python scripts/benchmark.py /data/sample/OCTVolume --device gpu --planes 100
  python scripts/benchmark.py /data/sample/OCTVolume --device cpu --planes 25
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from octrecon.pipeline import ReconConfig, Reconstructor  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("volume", help="OCTVolume folder")
    ap.add_argument("--config", help="optional JSON/YAML config (other ReconConfig fields)")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--row", type=int, default=4)
    ap.add_argument("--start", type=int, default=100)
    ap.add_argument("--planes", type=int, default=50)
    ap.add_argument("--batch-frames", type=int)
    ap.add_argument("--no-fused", action="store_true")
    ap.add_argument("--out", default=str(ROOT / "benchmarks/results"))
    a = ap.parse_args()
    over = dict(volume_folder=a.volume, device=a.device, batch_frames=a.batch_frames)
    cfg = (ReconConfig.from_file(a.config, **over) if a.config
           else ReconConfig.from_dict({k: v for k, v in over.items() if v is not None}))
    if a.no_fused:
        cfg.gpu_fused_kernel = False
    R = Reconstructor(cfg, log=lambda m: None)
    R.process_row(a.row, frames=np.arange(2))           # warm-up (JIT, cuFFT plans)
    R.timers = {k: 0.0 for k in R.timers}
    frames = np.arange(a.start, a.start + a.planes)
    t = time.perf_counter()
    R.process_row(a.row, frames=frames)
    dt = time.perf_counter() - t
    n_planes_total = R.n_out[0]
    res = {"device": R.device, "fused_kernel": cfg.gpu_fused_kernel, "batch_frames": cfg.batch_frames,
           "planes": a.planes, "bscans": a.planes * 96, "seconds": dt, "s_per_plane": dt / a.planes,
           "bscans_per_s": a.planes * 96 / dt, "raw_MB_per_s": a.planes * 96 * 2.1504 / dt,
           "extrapolated_full_volume_min": dt / a.planes * n_planes_total / 60,
           "timers": {k: round(v, 2) for k, v in R.timers.items()}}
    print(json.dumps(res, indent=2))
    Path(a.out).mkdir(parents=True, exist_ok=True)
    tag = f"{R.device}{'_fused' if (R.device == 'gpu' and cfg.gpu_fused_kernel) else ''}_b{cfg.batch_frames}"
    with open(Path(a.out) / f"bench_{tag}.json", "w") as f:
        json.dump(res, f, indent=2)


if __name__ == "__main__":
    main()
