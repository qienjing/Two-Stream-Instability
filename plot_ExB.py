import numpy as np
import matplotlib.pyplot as plt

from backend import xp
from grid1D import Grid1D
from particles import Particles
from push_particle import push_particle   # 这里改成你实际的文件名

def unwrap_periodic(x_arr, Lx):
    """
    把周期边界下的 x(t) 展开成连续轨迹，方便和解析解对比。
    假设每次跳跃不会超过 Lx/2。
    """
    x_unwrap = x_arr.copy()
    for i in range(1, len(x_unwrap)):
        dx = x_unwrap[i] - x_unwrap[i-1]
        if dx > 0.5 * Lx:
            x_unwrap[i:] -= Lx
        elif dx < -0.5 * Lx:
            x_unwrap[i:] += Lx
    return x_unwrap

def test_1d_sine_E():
    # ---------- 参数设置 ----------
    Np   = 1       # 只测一个粒子就够了
    Lx   = 20.0    # 盒子足够大，避免快速跑出边界
    Nx   = 64      # Grid1D 的网格数（只是为了构造对象，E 不用插值）
    dt   = 0.01
    nstep = 4000   # 总步数
    E0   = 0.1     # 电场振幅
    omega = 0.4    # 电场角频率

    # 初始条件
    x0 = 5.0
    v0 = 0.2

    # ---------- 初始化 PIC 结构 ----------
    grid = Grid1D(Lx=Lx, Nx=Nx)
    parts = Particles(Np=Np, Lx=Lx, n0=1.0)

    # 粒子初始位置和速度（只用 vx 分量）
    parts.x[:] = x0
    parts.v[:] = 0.0
    parts.v[:, 0] = v0   # vx = v0

    # ---------- 保存轨迹 ----------
    ts = np.arange(nstep+1) * dt
    x_num = np.zeros(nstep+1)
    v_num = np.zeros(nstep+1)

    x_num[0] = float(parts.x[0])
    v_num[0] = float(parts.v[0,0])

    # ---------- 时间推进 ----------
    for n in range(nstep):
        # 这里取 t 中点（leapfrog），你也可以用 t_n 或 t_{n+1/2}，只要解析解一致
        t_mid = (n + 0.5) * dt

        Ex_val = E0 * np.sin(omega * t_mid)

        # 因为你是 1D3V+电静，场是粒子点上的 Ex_p, Ey_p, Ez_p, Bx_p, By_p, Bz_p
        Ex_p = xp.full(Np, Ex_val, dtype=xp.float64)
        Ey_p = xp.zeros(Np, dtype=xp.float64)
        Ez_p = xp.zeros(Np, dtype=xp.float64)
        Bx_p = xp.zeros(Np, dtype=xp.float64)
        By_p = xp.zeros(Np, dtype=xp.float64)
        Bz_p = xp.zeros(Np, dtype=xp.float64)

        push_particle(parts, grid, Ex_p, Ey_p, Ez_p, Bx_p, By_p, Bz_p, dt)

        x_num[n+1] = float(parts.x[0])
        v_num[n+1] = float(parts.v[0,0])

    # ---------- 解析解 ----------
    # dv/dt = -E0 sin(omega t), v(0)=v0
    # v(t) = v0 + (E0/omega) (cos(omega t) - 1)
    v_ana = v0 + (E0 / omega) * (np.cos(omega * ts) - 1.0)

    # x(t) = x0 + (v0 - E0/omega) t + (E0/omega^2) sin(omega t)
    x_ana = x0 + (v0 - E0 / omega) * ts + (E0 / omega**2) * np.sin(omega * ts)

    # 把数值 x 轨迹从周期边界“解包”，便于和解析解对比
    x_num_unwrap = unwrap_periodic(x_num, Lx)

    # ---------- 画图 ----------
    plt.figure(figsize=(8,4))
    plt.subplot(1,2,1)
    plt.title("Velocity vs Time")
    plt.plot(ts, v_num, label="v_num (Boris)")
    plt.plot(ts, v_ana, "--", label="v_analytic")
    plt.xlabel("t")
    plt.ylabel("v_x")
    plt.legend()

    plt.subplot(1,2,2)
    plt.title("Position vs Time (unwrapped)")
    plt.plot(ts, x_num_unwrap, label="x_num (Boris)")
    plt.plot(ts, x_ana, "--", label="x_analytic")
    plt.xlabel("t")
    plt.ylabel("x")
    plt.legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    test_1d_sine_E()
    # ---- 5. 转成 numpy 方便画图 ----
    ts_np = to_np(ts)
    xs_np = to_np(xs)
    vs_np = to_np(vs)

    # ---- 6. 解析解 ----
    v_ana = a * ts_np                      # v_x(t) = a t
    x_ana = 0.5 * a * ts_np ** 2           # x(t) = 0.5 a t^2

    # ---- 7. 画图：横着的两个子图 ----
    fig, axes = plt.subplots(1, 2, figsize=(8, 3))

    # 左：v_x(t)
    axes[0].plot(ts_np, vs_np, label="Boris (numerical)")
    axes[0].plot(ts_np, v_ana, "--", label="Analytical")
    axes[0].set_xlabel("t")
    axes[0].set_ylabel(r"$v_x$")
    axes[0].set_title("Velocity in uniform $E_x$")
    axes[0].legend()
    axes[0].grid(True)

    # 右：x(t)
    axes[1].plot(ts_np, xs_np, label="Boris (numerical)")
    axes[1].plot(ts_np, x_ana, "--", label="Analytical")
    axes[1].set_xlabel("t")
    axes[1].set_ylabel("x")
    axes[1].set_title("Position in uniform $E_x$")
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig("boris_uniformE_validation.png", dpi=200)
    plt.show()


if __name__ == "__main__":
    test_boris_uniform_E()