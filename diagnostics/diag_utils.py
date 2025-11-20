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

def list_spectrum_files(outdir="output"):
    """返回所有电场谱文件 spectrum_*.npz 的完整路径"""
    files = sorted(glob.glob(os.path.join(outdir, "spectrum_*.npz")))
    if not files:
        raise FileNotFoundError(f"No spectrum_*.npz found in {outdir}")
    return files

def load_spectrum(file_npz):
    """读取单个 npz 文件，返回 (k, Ek, Pk)"""
    d = np.load(file_npz)
    k = d["k"]; Ek = d["Ek"]; Pk = d["Pk"]
    return k, Ek, Pk

def detect_kmax(files, skip_dc=True, avg_frames=5):
    """自动检测主导模式索引及波数"""
    ks = None; acc = None
    for f in files[:min(len(files), avg_frames)]:
        k, Ek, Pk = load_spectrum(f)
        if ks is None: ks, acc = k, np.zeros_like(Pk)
        acc += Pk
    if skip_dc and len(acc) > 1:
        idx = 1 + np.argmax(acc[1:])
    else:
        idx = int(np.argmax(acc))
    return idx, ks[idx], ks

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
