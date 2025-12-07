import numpy as np
from main_dimensionless import Grid1D, solve_poisson_1d, xp, CUPY


# ----------------------------
# 自动转换工具
# ----------------------------
def to_xp(a):
    if CUPY:
        return xp.asarray(a)
    return a

def to_np(a):
    if CUPY:
        return xp.asnumpy(a)
    return a


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
    Lx = 2*np.pi

    grid = Grid1D(Lx, Nx)
    rho = to_xp(np.zeros(Nx))

    Ex = solve_poisson_1d(grid, rho)
    print("Test 1 (rho=0): max|Ex| = ", float(np.max(np.abs(to_np(Ex)))))


# ========================================
# Test 2: rho = cos(kx)
# ========================================
def test_single_mode(m=1):
    Nx = 256
    Lx = 2*np.pi

    grid = Grid1D(Lx, Nx)
    x = np.linspace(0, Lx, Nx, endpoint=False)

    k = 2*np.pi*m / Lx

    rho = np.cos(k * x)
    Ex_num = solve_poisson_1d(grid, to_xp(rho))
    Ex_num = to_np(Ex_num)

    Ex_true = (1.0/k) * np.sin(k * x)

    print(f"Test 2 (m={m}): L2 error = ", L2(Ex_true, Ex_num))


# ========================================
# Test 3: 高频模式
# ========================================
def test_high_k():
    test_single_mode(10)


# ========================================
# Test 4: 随机 rho
# ========================================
def test_random():
    Nx = 256
    Lx = 2*np.pi

    grid = Grid1D(Lx, Nx)

    rho = np.random.randn(Nx)
    rho_xp = to_xp(rho)

    Ex_num = solve_poisson_1d(grid, rho_xp)
    Ex_num = to_np(Ex_num)

    # ---------- 解析解 (同样使用 xp 后端计算) ----------
    rho_k = xp.fft.rfft(to_xp(rho))

    kd = grid.kd
    k = grid.k

    phi_k = xp.zeros_like(rho_k)
    phi_k[1:] = -rho_k[1:] / (kd[1:] ** 2)

    Ex_true_k = -1j * k * phi_k
    Ex_true = xp.fft.irfft(Ex_true_k, n=Nx)
    Ex_true = to_np(Ex_true)

    print("Test 4 (random rho): L2 error = ", L2(Ex_true, Ex_num))


# ========================================
# Run
# ========================================
if __name__ == "__main__":
    print("===== Testing Poisson solver =====")
    print("Backend:", "CUPY" if CUPY else "NUMPY")
    print()

    test_zero()
    test_single_mode(1)
    test_single_mode(3)
    test_high_k()
    test_random()
