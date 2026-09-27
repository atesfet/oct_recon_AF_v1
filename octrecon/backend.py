"""Array backend selection.

  cpu        NumPy + Numba + multithreaded pocketfft
  gpu        NVIDIA GPU via CuPy (fused CUDA kernel, cuFFT)                 -> name "gpu"
             or, on Apple silicon, the GPU via PyTorch Metal/MPS            -> name "mps"
  mps        force the Apple GPU (PyTorch MPS)
  torch-cpu  the PyTorch backend on the CPU (used to validate the MPS code path anywhere)
  auto       CUDA if usable, else MPS if usable (and its self-test passes), else CPU

Every numerical routine takes an ``xp`` so the same pipeline runs on all backends.
"""
from __future__ import annotations

import os

import numpy as np

# let PyTorch run ops that MPS does not implement on the CPU instead of failing
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

_MPS_STATUS: tuple[bool, str] | None = None


def gpu_available() -> bool:
    """NVIDIA GPU usable through CuPy."""
    try:
        import cupy as cp
        cp.cuda.runtime.getDeviceCount()
        cp.zeros(1).sum()  # forces context creation; fails on a broken driver
        return True
    except Exception:
        return False


def mps_status() -> tuple[bool, str]:
    """(usable, message) for the Apple-silicon GPU via PyTorch MPS (cached)."""
    global _MPS_STATUS
    if _MPS_STATUS is None:
        try:
            import torch
            if not torch.backends.mps.is_available():
                built = torch.backends.mps.is_built()
                _MPS_STATUS = (False, "PyTorch MPS not available on this machine" if built
                               else "PyTorch was built without MPS support")
            else:
                from .core.torch_backend import TorchXP, selftest
                ok, msg = selftest(TorchXP("mps"))
                _MPS_STATUS = (ok, "Apple GPU (Metal/MPS) self-test passed" if ok
                               else f"MPS self-test failed ({msg}); using CPU")
        except ImportError:
            _MPS_STATUS = (False, "PyTorch is not installed")
        except Exception as e:  # noqa: BLE001
            _MPS_STATUS = (False, f"MPS unavailable: {type(e).__name__}: {e}")
    return _MPS_STATUS


def mps_available() -> bool:
    return mps_status()[0]


def get_xp(device: str = "auto"):
    """Return (xp, device_name) with device_name in {'gpu', 'mps', 'torch-cpu', 'cpu'}."""
    if device not in ("auto", "gpu", "cpu", "mps", "torch-cpu"):
        raise ValueError(f"device must be auto|gpu|cpu|mps, got {device!r}")
    if device == "cpu":
        return np, "cpu"
    if device == "torch-cpu":
        from .core.torch_backend import TorchXP
        return TorchXP("cpu"), "torch-cpu"
    if device in ("auto", "gpu") and gpu_available():
        import cupy as cp
        return cp, "gpu"
    if device in ("auto", "gpu", "mps") and mps_available():
        from .core.torch_backend import TorchXP
        return TorchXP("mps"), "mps"
    if device == "auto":
        return np, "cpu"
    if device == "mps":
        raise RuntimeError(f"Apple GPU requested but not usable: {mps_status()[1]}")
    raise RuntimeError(
        "GPU requested but no usable GPU was found. NVIDIA: CUDA could not be initialised "
        "(after a laptop suspend this usually needs a reboot or "
        "`sudo rmmod nvidia_uvm && sudo modprobe nvidia_uvm`). Apple silicon: " + mps_status()[1])


def is_torch(xp) -> bool:
    return hasattr(xp, "torch") and hasattr(xp, "device")


def to_numpy(a):
    if isinstance(a, np.ndarray):
        return a
    if hasattr(a, "detach"):          # torch tensor
        return a.detach().cpu().numpy()
    return a.get()                    # cupy


def synchronize(xp):
    if xp is np:
        return
    if is_torch(xp):
        xp.synchronize()
    else:
        xp.cuda.Stream.null.synchronize()
