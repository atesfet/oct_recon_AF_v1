"""octrecon: GPU-accelerated reconstruction of tiled Thorlabs OCT scans (port of myOCT)."""
import os as _os

# avoid oversubscription between numba / pocketfft threads and OpenBLAS (measured 26s -> 3s stitch)
_os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
