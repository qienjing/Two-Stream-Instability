# -*- coding: utf-8 -*-
"""
backend.py
Manage NumPy/CuPy backend selection, FFT, random seeds, and to_np/to_xp helpers.
"""

import math

USE_cupy = True     # Legacy flag (still supported)
FORCE_DEVICE = "gpu"   # "cpu", "gpu", or "auto"
INIT_DEVICE = "gpu"   # internal use only: "cpu" or "gpu"
SEED = 12345

# ============================================================
# Backend selection logic for CPU / GPU
# ============================================================

def _force_cpu_backend():
    import numpy as np
    return {
        "xp": np,
        "fft": np.fft,
        "scatter_add": None,
        "CUPY": False
    }

def _try_gpu_backend():
    import cupy as cp
    from cupyx import scatter_add as cpx_scatter_add
    cp.random.seed(SEED)
    cp.cuda.Stream.null.synchronize()
    return {
        "xp": cp,
        "fft": cp.fft,
        "scatter_add": cpx_scatter_add,
        "CUPY": True
    }

# ----------- Device selection flow -----------
backend = None

if FORCE_DEVICE.lower() == "cpu":
    backend = _force_cpu_backend()

elif FORCE_DEVICE.lower() == "gpu":
    try:
        backend = _try_gpu_backend()
    except Exception as e:
        raise RuntimeError("Forced GPU mode but CuPy is unavailable: " + str(e))

elif FORCE_DEVICE.lower() == "auto":
    # Original auto mode: prefer GPU, fall back to CPU on failure
    try:
        backend = _try_gpu_backend()
    except Exception:
        backend = _force_cpu_backend()

else:
    raise ValueError('FORCE_DEVICE must be "cpu", "gpu", or "auto"')

# Unpack
xp = backend["xp"]
fft = backend["fft"]
scatter_add = backend["scatter_add"]
CUPY = backend["CUPY"]

# CPU RNG still needs initialization
if not CUPY:
    xp.random.seed(SEED)

# ============================================================
# Utility functions
# ============================================================

def to_xp(a):
    """Convert numpy → cupy if needed"""
    return xp.asarray(a) if CUPY else a

def to_np(a):
    """Convert cupy → numpy if needed"""
    return xp.asnumpy(a) if CUPY else a

def zeros_like_shape(shape, dtype=xp.float64):
    return xp.zeros(shape, dtype=dtype)

def roll(a, shift, axis):
    return xp.roll(a, shift, axis=axis)
