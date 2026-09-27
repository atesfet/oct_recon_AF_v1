"""CuPy fused CUDA kernel for the spectral pre-FFT stage (GPU counterpart of
cpu_kernels.fused_pre_fft). One thread per output spectral sample.

The raw input may be int16 (Thorlabs OCITY / SRR), uint16 (Wasatch) or float32
(after AScanBinning / synthetic frame-mean apodization); one kernel is compiled
per input type."""
import numpy as np
import cupy as cp

_SRC = r'''
#include <cupy/complex.cuh>
extern "C" __global__
void fused_pre_fft(const RAW_T* __restrict__ raw, const float* __restrict__ apod,
                   const float* __restrict__ mean, const long long* __restrict__ idx,
                   const float* __restrict__ w, const complex<float>* __restrict__ win,
                   complex<float>* __restrict__ out,
                   int B, int interf, int apod_size, int N, int T)
{
    long long nX = interf - apod_size;
    long long total = (long long)B * nX * N;
    for (long long p = blockIdx.x * (long long)blockDim.x + threadIdx.x; p < total;
         p += (long long)blockDim.x * gridDim.x) {
        int i = p % N;
        long long line = p / N;          // b*nX + x
        int b = line / nX;
        int x = line - (long long)b * nX;
        const RAW_T* r = raw + ((long long)b * interf + apod_size + x) * N;
        const float* a = apod + (long long)b * N;
        float m = mean[line];
        float acc = 0.f;
        for (int t = 0; t < T; ++t) {
            long long j = idx[(long long)t * N + i];
            acc += w[(long long)t * N + i] * (((float)r[j] - a[j]) - m);
        }
        out[p] = (acc + m) * win[i];
    }
}
'''
_CTYPES = {np.dtype(np.int16): "short", np.dtype(np.uint16): "unsigned short",
           np.dtype(np.float32): "float"}
SUPPORTED_RAW_DTYPES = tuple(_CTYPES)
_kernels = {}


def _kernel_for(dtype):
    dt = np.dtype(dtype)
    k = _kernels.get(dt)
    if k is None:
        src = "#define RAW_T " + _CTYPES[dt] + "\n" + _SRC
        k = _kernels[dt] = cp.RawKernel(src, "fused_pre_fft")
    return k


def fused_pre_fft(raw, apod_size, idx, w, window):
    """raw (B, interf, N) int16/uint16/float32 on device -> complex64 (B, nX, N)."""
    B, interf, N = raw.shape
    rawf_apod = raw[:, :apod_size, :].astype(cp.float32).mean(axis=1)            # (B, N)
    # per-A-line mean of apod-subtracted spectrum = mean(raw line) - mean(apod)
    line_mean = raw[:, apod_size:, :].astype(cp.float32).mean(axis=2) - rawf_apod.mean(axis=1, keepdims=True)
    out = cp.empty((B, interf - apod_size, N), cp.complex64)
    total = out.size
    threads = 256
    blocks = min((total + threads - 1) // threads, 65535 * 4)
    _kernel_for(raw.dtype)((blocks,), (threads,),
                           (cp.ascontiguousarray(raw), rawf_apod, cp.ascontiguousarray(line_mean),
                            idx, w, window, out, cp.int32(B), cp.int32(interf), cp.int32(apod_size),
                            cp.int32(N), cp.int32(idx.shape[0])))
    return out
