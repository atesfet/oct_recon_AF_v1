"""Install GPU support (CuPy) into the current environment if an NVIDIA GPU driver is present.

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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conda")
    ap.add_argument("--env")
    ap.add_argument("--pip", action="store_true")
    a = ap.parse_args()
    if sys.platform == "darwin":
        print("[setup_gpu] macOS: no CUDA support, CPU mode only."); return
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
