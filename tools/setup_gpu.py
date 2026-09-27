"""Install GPU support into the current environment.

  * NVIDIA GPU (Linux/Windows): CuPy, with the CUDA toolkit matched to the driver.
  * macOS (Apple silicon / Metal GPU): PyTorch, whose MPS backend drives the Apple GPU.

Called by the launchers:
    python tools/setup_gpu.py --conda <path-to-conda> --env <name>     (conda mode)
    python tools/setup_gpu.py --pip                                     (venv/pip mode)
The CUDA toolkit version is matched to the driver (nvidia-smi "CUDA Version"), because a
newer CUDA runtime than the driver supports fails at run time.
"""
import argparse
import re
import shutil
import subprocess
import sys


def driver_cuda_version():
    exe = shutil.which("nvidia-smi")
    if not exe:
        return None
    try:
        out = subprocess.run([exe], capture_output=True, text=True, timeout=20).stdout
    except Exception:
        return None
    m = re.search(r"CUDA Version:\s*([0-9]+)\.([0-9]+)", out)
    return (int(m.group(1)), int(m.group(2))) if m else None


def have_cupy():
    try:
        import cupy  # noqa: F401
        return True
    except Exception:
        return False


def have_torch_mps():
    try:
        import torch
        return torch.backends.mps.is_available()
    except Exception:
        return None          # torch missing


def setup_mac(a):
    st = have_torch_mps()
    if st:
        print("[setup_gpu] PyTorch with Apple GPU (MPS) support is installed."); return
    if st is False:
        print("[setup_gpu] PyTorch is installed but MPS is not available on this Mac "
              "(needs Apple silicon or a Metal GPU and macOS 13+) -> CPU mode."); return
    print("[setup_gpu] macOS: installing PyTorch for Apple-GPU (Metal/MPS) acceleration ...", flush=True)
    try:
        if a.conda:
            subprocess.check_call([a.conda, "install", "-y", "-n", a.env, "-c", "conda-forge", "pytorch"])
        else:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements-mac.txt"])
        print("[setup_gpu] done.")
    except subprocess.CalledProcessError as e:
        print(f"[setup_gpu] PyTorch could not be installed ({e}); continuing in CPU mode.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conda")
    ap.add_argument("--env")
    ap.add_argument("--pip", action="store_true")
    a = ap.parse_args()
    if sys.platform == "darwin":
        return setup_mac(a)
    ver = driver_cuda_version()
    if ver is None:
        print("[setup_gpu] no NVIDIA driver found (nvidia-smi) -> CPU mode only."); return
    if have_cupy():
        print(f"[setup_gpu] CuPy already installed (driver CUDA {ver[0]}.{ver[1]})."); return
    print(f"[setup_gpu] NVIDIA driver supports CUDA {ver[0]}.{ver[1]} -> installing CuPy ...", flush=True)
    try:
        if a.conda:
            subprocess.check_call([a.conda, "install", "-y", "-n", a.env, "-c", "conda-forge",
                                   "cupy", f"cuda-version={ver[0]}.{ver[1]}"])
        else:
            if ver[0] != 12:
                print("[setup_gpu] pip mode supports CUDA 12 drivers; install CuPy manually "
                      "(https://docs.cupy.dev/en/stable/install.html)."); return
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements-gpu.txt"])
        print("[setup_gpu] done.")
    except subprocess.CalledProcessError as e:
        print(f"[setup_gpu] GPU packages could not be installed ({e}); continuing in CPU mode.")


if __name__ == "__main__":
    main()
