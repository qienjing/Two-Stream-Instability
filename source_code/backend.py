# -*- coding: utf-8 -*-
"""
backend.py
统一管理 NumPy/CuPy 后端、FFT、随机种子、to_np/to_xp 工具函数
"""

import math

USE_CUPY = True
SEED = 12345

try:
    if USE_CUPY:
        import cupy as cp
        from cupyx import scatter_add as cpx_scatter_add
        xp = cp
        fft = cp.fft
        scatter_add = cpx_scatter_add
        CUPY = True

        cp.random.seed(SEED)
        cp.cuda.Stream.null.synchronize()
    else:
        raise ImportError

except ImportError:
    import numpy as np
    xp = np
    import numpy.fft as fft
    scatter_add = None
    CUPY = False

    np.random.seed(SEED)


# ============== 工具函数 ==============

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
