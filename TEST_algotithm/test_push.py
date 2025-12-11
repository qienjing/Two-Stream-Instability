import numpy as np
import matplotlib.pyplot as plt
import sys, os

# 加载上一级目录模块
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from backend import xp, to_np
from grid1D import Grid1D
from particles import Particles
from push_particle import push_particle


def unwrap_periodic(x_arr, Lx):
    x_unwrap = x_arr.copy()
    for i in range(1, len(x_unwrap)):
        dx = x_unwrap[i] - x_unwrap[i-1]
        if dx > 0.5 * Lx:
            x_unwrap[i:] -= Lx
        elif dx < -0.5 * Lx:
            x_unwrap[i:] += Lx
    return x_unwrap


def test_1d_sine_E():
    # ---------- 参数 ----------
    Np   = 1
    Lx   = 20.0
    Nx   = 64
    dt   = 0.01
    nstep = 4000
    E0   = 0.1
    omega = 0.4

    # 初始条件
    x0 = 5.0
    v0 = 0.2

    # ---------- 初始化 ----------
    grid = Grid1D(Lx=Lx, Nx=Nx)
    parts = Particles(Np=Np, Lx=Lx, n0=1.0)

    parts.x[:] = x0
    parts.v[:] = 0.0
    parts.v[:, 0] = v0  # vx = v0

    # ---------- 保存轨迹 ----------
    ts = np.arange(nstep+1) * dt
    x_num = np.zeros(nstep+1)
    v_num = np.zeros(nstep+1)

    x_num[0] = float(parts.x[0])
    v_num[0] = float(parts.v[0, 0])

    # ---------- 主循环 ----------
    for n in range(nstep):
        t_mid = (n + 0.5) * dt
        Ex_val = E0 * np.sin(omega * t_mid)

        Ex_p = xp.full(Np, Ex_val)
        Ey_p = xp.zeros(Np)
        Ez_p = xp.zeros(Np)
        Bx_p = xp.zeros(Np)
        By_p = xp.zeros(Np)
        Bz_p = xp.zeros(Np)

        push_particle(parts, grid, Ex_p, Ey_p, Ez_p, Bx_p, By_p, Bz_p, dt)

        x_num[n+1] = float(parts.x[0])
        v_num[n+1] = float(parts.v[0, 0])

    # ---------- 解析解 ----------
    v_ana = v0 + (E0 / omega) * (np.cos(omega * ts) - 1.0)
    x_ana = x0 + (v0 - E0 / omega) * ts + (E0 / omega**2) * np.sin(omega * ts)

    x_unwrap = unwrap_periodic(x_num, Lx)

    # ---------- 绘图 ----------
    plt.figure(figsize=(8, 4))

    plt.subplot(1, 2, 1)
    plt.title("Velocity vs Time")
    plt.plot(ts, v_num, label="numerical")
    plt.plot(ts, v_ana, "--", label="analytic")
    plt.xlabel("t"); plt.ylabel("vx"); plt.legend()

    plt.subplot(1, 2, 2)
    plt.title("Position vs Time (unwrapped)")
    plt.plot(ts, x_unwrap, label="numerical")
    plt.plot(ts, x_ana, "--", label="analytic")
    plt.xlabel("t"); plt.ylabel("x"); plt.legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    test_1d_sine_E()
