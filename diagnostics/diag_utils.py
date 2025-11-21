# diagnostics/diag_utils.py
import os, glob, numpy as np

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path

def load_energy(outdir="output"):
    path = os.path.join(outdir, "energy.txt")
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found.")
    return np.loadtxt(path)

def load_modes_history(outdir="output"):
    path = os.path.join(outdir, "modes_history.txt")
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found.")
    return np.loadtxt(path)

def robust_linear_fit(t, y, frac=1/3, min_pts=6):
    """在前 frac 部分时间窗口内线性拟合 y ~ a + b t"""
    n = max(int(len(t)*frac), min_pts)
    n = min(n, len(t))
    if n < 2: return np.nan, np.nan
    ok = np.isfinite(t[:n]) & np.isfinite(y[:n])
    if ok.sum() < 2: return np.nan, np.nan
    a, b = np.polyfit(t[:n][ok], y[:n][ok], 1)
    return a, b

def unwrap_phase(ph):
    return np.unwrap(ph)

def save_txt(path, text):
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
