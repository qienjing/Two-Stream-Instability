# -*- coding: utf-8 -*-
"""
backend.py
统一管理 NumPy/CuPy 后端、FFT、随机种子、to_np/to_xp 工具函数
"""

import math

USE_cupy = True     # 原来的参数（仍支持）
FORCE_DEVICE = "gpu"   # "cpu", "gpu", or "auto"
INIT_DEVICE = "gpu"   # internal use only: "cpu" or "gpu"
SEED = 12345

# ============================================================
# 选择 CPU / GPU 后端逻辑
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

# ----------- 设备选择流程 -----------
backend = None

if FORCE_DEVICE.lower() == "cpu":
    backend = _force_cpu_backend()

elif FORCE_DEVICE.lower() == "gpu":
    try:
        backend = _try_gpu_backend()
    except Exception as e:
        raise RuntimeError("强制 GPU 模式，但 CuPy 不可用: " + str(e))

elif FORCE_DEVICE.lower() == "auto":
    # 原 auto：优先 GPU，失败则 CPU
    try:
        backend = _try_gpu_backend()
    except Exception:
        backend = _force_cpu_backend()

else:
    raise ValueError('FORCE_DEVICE 必须为 "cpu", "gpu", 或 "auto"')

# 解包
xp = backend["xp"]
fft = backend["fft"]
scatter_add = backend["scatter_add"]
CUPY = backend["CUPY"]

# CPU RNG 仍然需要初始化
if not CUPY:
    xp.random.seed(SEED)

# ============================================================
# 工具函数
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
