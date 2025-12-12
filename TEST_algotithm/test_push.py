import numpy as np
import matplotlib.pyplot as plt
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import source_code
from source_code.backend import xp, to_np
from source_code.grid1D import Grid1D
from source_code.particles import Particles
from source_code.push_particle import push_particle


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
    Np   = 1
    Lx   = 20.0
    Nx   = 64
    dt   = 0.01
    nstep = 4000
    E0   = 0.1
    omega = 0.4


    x0 = 5.0
    v0 = 0.2

    grid = Grid1D(Lx=Lx, Nx=Nx)
    parts = Particles(Np=Np, Lx=Lx, n0=1.0)

    parts.x[:] = x0
    parts.v[:] = 0.0
    parts.v[:, 0] = v0  # vx = v0

    ts = np.arange(nstep+1) * dt
    x_num = np.zeros(nstep+1)
    v_num = np.zeros(nstep+1)

    x_num[0] = float(parts.x[0])
    v_num[0] = float(parts.v[0, 0])

    # main loop
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

    # the analytic solution
    v_ana = v0 + (E0 / omega) * (np.cos(omega * ts) - 1.0)
    x_ana = x0 + (v0 - E0 / omega) * ts + (E0 / omega**2) * np.sin(omega * ts)

    x_unwrap = unwrap_periodic(x_num, Lx)

    plt.figure(figsize=(8, 4), dpi=100)
    # Global font settings
    title_size = 18
    label_size = 14
    tick_size = 14
    legend_size = 14

    # ------------------------
    plt.subplot(1, 2, 1)
    plt.title("Velocity vs Time", fontsize=title_size)
    plt.plot(ts, v_num, label="numerical", lw=3)
    plt.plot(ts, v_ana, "--", label="analytic", lw=3)
    plt.xlabel("t", fontsize=label_size)
    plt.ylabel("vx", fontsize=label_size)
    plt.xticks(fontsize=tick_size)
    plt.yticks(fontsize=tick_size)
    plt.legend(fontsize=legend_size)

    # ------------------------
    plt.subplot(1, 2, 2)
    plt.title("Position vs Time", fontsize=title_size)
    plt.plot(ts, x_unwrap, label="numerical", lw=3)
    plt.plot(ts, x_ana, "--", label="analytic", lw=3)
    plt.xlabel("t", fontsize=label_size)
    plt.ylabel("x", fontsize=label_size)
    plt.xticks(fontsize=tick_size)
    plt.yticks(fontsize=tick_size)
    plt.legend(fontsize=legend_size)

    plt.tight_layout()
    plt.show()



if __name__ == "__main__":
    test_1d_sine_E()
