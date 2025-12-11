import numpy as np
import os, sys

# -------------------------------------------------
# 让 test 可以从根目录找到 source_code 包
# -------------------------------------------------
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.append(ROOT)

from source_code.grid1D import Grid1D
from source_code.solve_poisson_1d import solve_poisson_1d
from source_code.backend import xp, CUPY, to_xp, to_np


# ----------------------------
# L2 误差
# ----------------------------
def L2(a, b):
    a = to_np(a)
    b = to_np(b)
    return np.sqrt(np.mean((a - b) ** 2))


# ========================================
# Test 1: rho = 0 → E = 0
# ========================================
def test_zero():
    Nx = 128
    Lx = 2 * np.pi

    grid = Grid1D(Lx, Nx)
    rho = to_xp(np.zeros(Nx))

    Ex, phi = solve_poisson_1d(grid, rho)
    Ex = to_np(Ex)

    print("Test 1 (rho=0): max|Ex| =", float(np.max(np.abs(Ex))))


# ========================================
# Test 2: rho = cos(kx)
# ========================================
def test_single_mode(m=1):
    Nx = 256
    Lx = 2 * np.pi

    grid = Grid1D(Lx, Nx)
    x = np.linspace(0, Lx, Nx, endpoint=False)

    k = 2 * np.pi * m / Lx

    rho = np.cos(k * x)
    Ex_num, phi = solve_poisson_1d(grid, to_xp(rho))
    Ex_num = to_np(Ex_num)

    # ---- 解析解 ----
    Ex_true = (1.0 / k) * np.sin(k * x)

    print(f"Test 2 (m={m}): L2 error =", L2(Ex_true, Ex_num))


# ========================================
# Test 3: high-k mode
# ========================================
def test_high_k():
    test_single_mode(10)


# ========================================
# Test 4: random rho
# ========================================
def test_random():
    Nx = 256
    Lx = 2 * np.pi
    grid = Grid1D(Lx, Nx)

    rho = np.random.randn(Nx)
    rho_xp = to_xp(rho)

    Ex_num, phi_num = solve_poisson_1d(grid, rho_xp)
    Ex_num = to_np(Ex_num)

    # ---- 构造解析解 (与 solver 一致，无 kd) ----
    rho_k = xp.fft.rfft(to_xp(rho))
    k = grid.k                          # <<< 使用 k，不使用 kd

    phi_k = xp.zeros_like(rho_k)
    phi_k[1:] = rho_k[1:] / (k[1:]**2)

    Ex_true_k = -1j * k * phi_k
    Ex_true = xp.fft.irfft(Ex_true_k, n=Nx)
    Ex_true = to_np(Ex_true)

    print("Test 4 (random rho): L2 error =", L2(Ex_true, Ex_num))


# ========================================
# Run
# ========================================
if __name__ == "__main__":
    print("===== Testing Poisson solver =====")
    print("Backend:", "CuPy (GPU)" if CUPY else "NumPy (CPU)")
    print()

    test_zero()
    test_single_mode(1)
    test_single_mode(3)
    test_high_k()
    test_random()
