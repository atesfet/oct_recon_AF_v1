"""Array backend selection (NumPy on CPU, CuPy on GPU).

Every numerical routine in octrecon takes an ``xp`` module so the exact same
code path runs on CPU (validation / fallback) and GPU (production).
"""
from __future__ import annotations

import numpy as np


def gpu_available() -> bool:
    try:
        import cupy as cp
        cp.cuda.runtime.getDeviceCount()
        cp.zeros(1).sum()  # forces context creation; fails on a broken driver
        return True
    except Exception:
        return False


def get_xp(device: str = "auto"):
    """Return (xp, device_name). device in {'auto', 'gpu', 'cpu'}."""
    if device not in ("auto", "gpu", "cpu"):
        raise ValueError(f"device must be auto|gpu|cpu, got {device!r}")
    if device in ("auto", "gpu"):
        if gpu_available():
            import cupy as cp
            return cp, "gpu"
        if device == "gpu":
            raise RuntimeError(
                "GPU requested but CUDA is not usable (cuInit failed?). "
                "After a laptop suspend this usually needs a reboot or "
                "`sudo rmmod nvidia_uvm && sudo modprobe nvidia_uvm`.")
    return np, "cpu"


def to_numpy(a):
    if isinstance(a, np.ndarray):
        return a
    return a.get()


def synchronize(xp):
    if xp is not np:
        xp.cuda.Stream.null.synchronize()
