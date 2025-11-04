# -*- coding: utf-8 -*-
"""
1D3V Electrostatic PIC for Two-Stream Instability with Modular Diagnostics
- Space: 1D (x), Periodic
- Velocity: 3 components (vx, vy, vz) -> EM-ready
- Fields: Electrostatic (solve Poisson -> Ex). EM hooks kept for future.
- Pusher: Boris (works for B=0)
- GPU: CuPy if available, otherwise NumPy

Outputs (under ./output):
  - energy.txt                     # [step, W_E, W_K, W_T]
  - growth.txt                     # summary: gamma_total, gamma_kmax, omega_kmax, kmax
  - spectrum_00000.npz ...         # {'k': k, 'Ek': Ek_complex, 'Pk': |Ek|^2}
  - phase_00000.npz ...            # {'H': hist2d, 'xedges':..., 'vedges':...}

Author: (you)
"""

import os, math, glob

# ---------------------------
# Backend selection (GPU/CPU)
# ---------------------------
USE_CUPY = True
try:
    if USE_CUPY:
        import cupy as cp
        from cupyx import scatter_add as cpx_scatter_add
        xp = cp
        fft = cp.fft
        scatter_add = cpx_scatter_add
        CUPY = True
    else:
        raise ImportError
except Exception:
    import numpy as np
    xp = np
    import numpy.fft as fft
    scatter_add = None
    CUPY = False

# ---------------------------
# Utilities
# ---------------------------
def to_xp(a):
    return xp.asarray(a) if CUPY else a

def to_np(a):
    return xp.asnumpy(a) if CUPY else a

def zeros_like_shape(shape, dtype=xp.float32):
    return xp.zeros(shape, dtype=dtype)

def roll(a, shift, axis):
    return xp.roll(a, shift, axis=axis)

# ---------------------------
# Grid
# ---------------------------
class Grid1D:
    def __init__(self, Lx, Nx, dtype=xp.float32):
        self.Lx = float(Lx) # total length in x
        self.Nx = int(Nx) # number of grid points
        self.dx = self.Lx / self.Nx
        self.dtype = dtype
        # rFFT wave numbers
        m = xp.arange(self.Nx//2 + 1, dtype=dtype)
        self.k = (2.0 * math.pi / self.Lx) * m  # shape (Nx/2+1,)

# ---------------------------
# Fields container (ES now, EM ready)
# ---------------------------
class Fields:
    def __init__(self, grid: Grid1D, dtype=xp.float32):
        Nx = grid.Nx
        self.Ex = zeros_like_shape(Nx, dtype=dtype)
        self.Ey = zeros_like_shape(Nx, dtype=dtype)
        self.Ez = zeros_like_shape(Nx, dtype=dtype)
        self.Bx = zeros_like_shape(Nx, dtype=dtype)
        self.By = zeros_like_shape(Nx, dtype=dtype)
        self.Bz = zeros_like_shape(Nx, dtype=dtype)

# ---------------------------
# Particles (1D position, 3V velocity)
# ---------------------------
class Particles:
    def __init__(self, Np, q=-1.0, m=1.0, dtype=xp.float32):
        self.Np = int(Np) # number of total particles 
        self.q  = dtype(q)
        self.m  = dtype(m)
        self.x  = zeros_like_shape(self.Np, dtype=dtype)      # x in [0,Lx)
        self.v  = zeros_like_shape((self.Np,3), dtype=dtype)  # vx,vy,vz

# ---------------------------
# Deposition & Gather (CIC)
# ---------------------------
## Deposit charge density from particles to grid
def deposit_charge_CIC(grid: Grid1D, parts: Particles, rho_out):
    Nx, dx = grid.Nx, grid.dx
    x = parts.x
    q = parts.q
    Np = parts.Np

    xc = x / dx # grid index
    iL = xp.floor(xc).astype(xp.int32) # left grid point index
    x_incell = xc - iL # position in cell [0,1)
    iR = (iL + 1) % Nx

    wL = 1.0 - x_incell # weight of a particle on left grid point
    wR = x_incell
    rho_out.fill(0.0) # reset
    # charge per length per particle contribution
    contrib_L = q * wL / dx
    contrib_R = q * wR / dx

    if CUPY:
        scatter_add(rho_out, iL, contrib_L)
        scatter_add(rho_out, iR, contrib_R)
    else:
        xp.add.at(rho_out, iL, contrib_L)
        xp.add.at(rho_out, iR, contrib_R)

# Gather field from grid to particle positions
def gather_CIC_Ex(grid: Grid1D, Ex, x_particles):
    Nx, dx = grid.Nx, grid.dx
    xc = x_particles / dx
    iL = xp.floor(xc).astype(xp.int32)
    x_incell = xc - iL
    iR = (iL + 1) % Nx
    wL = 1.0 - x_incell
    wR = x_incell
    return wL * Ex[iL] + wR * Ex[iR]

# ---------------------------
# Poisson solver (periodic)
# ---------------------------
def solve_poisson_1d(grid: Grid1D, rho, eps0=1.0):
    Nx = grid.Nx
    k = grid.k
    rho_k = fft.rfft(rho)
    phi_k = xp.zeros_like(rho_k, dtype=rho_k.dtype)
    phi_k[1:] = -rho_k[1:] / (eps0 * (k[1:]**2))
    Ex_k = -1j * k * phi_k
    Ex = fft.irfft(Ex_k, n=Nx).astype(rho.dtype)
    return Ex

# ---------------------------
# Push particles (non-relativistic)
# ---------------------------
def push_velocity_boris(parts: Particles, Ex_p, Ey_p, Ez_p, Bx_p, By_p, Bz_p, dt):
    q, m = parts.q, parts.m
    v = parts.v

    h = (q * dt) / (2.0 * m)
    vmx = v[:,0] + h*Ex_p
    vmy = v[:,1] + h*Ey_p
    vmz = v[:,2] + h*Ez_p

    tx = (q * dt / m) * 0.5 * Bx_p
    ty = (q * dt / m) * 0.5 * By_p
    tz = (q * dt / m) * 0.5 * Bz_p

    vpx = vmx + (vmy*tz - vmz*ty)
    vpy = vmy + (vmz*tx - vmx*tz)
    vpz = vmz + (vmx*ty - vmy*tx)

    t2 = tx*tx + ty*ty + tz*tz
    s  = 2.0 / (1.0 + t2)
    sx, sy, sz = s*tx, s*ty, s*tz

    vpx2 = vmx + (vpy*sz - vpz*sy)
    vpy2 = vmy + (vpz*sx - vpx*sz)
    vpz2 = vmz + (vpx*sy - vpy*sx)

    v[:,0] = vpx2 + h*Ex_p
    v[:,1] = vpy2 + h*Ey_p
    v[:,2] = vpz2 + h*Ez_p

def push_particle(parts: Particles, grid: Grid1D, dt):
    parts.x += parts.v[:,0] * dt
    Lx = grid.Lx
    parts.x -= xp.floor(parts.x / Lx) * Lx

# ---------------------------
# Diagnostics (modular)
# ---------------------------
import numpy as _np

class Diagnostics:
    def __init__(self, grid: Grid1D, outdir="output", diag_interval=50,
                 phase_bins_x=None, phase_bins_v=100, phase_v_range=(-4.0, 4.0)):
        self.grid = grid
        self.outdir = outdir
        self.diag_interval = int(diag_interval)
        os.makedirs(outdir, exist_ok=True)

        # Energy time series
        self.energy_data = []  # rows: [step, W_E, W_K, W_T]

        # Spectrum tracking
        self.k_axis = (2*_np.pi/grid.Lx) * _np.arange(grid.Nx//2 + 1)
        self.kmax_index = None
        self.kmax_list_steps = []
        self.kmax_amps = []   # |E_kmax|
        self.kmax_phis = []   # arg(E_kmax) for frequency
        self.total_amps = []  # sqrt(sum Ex^2) ~ |E|_total

        # Phase space histogram setup
        self.phase_bins_x = int(grid.Nx // 4) if phase_bins_x is None else int(phase_bins_x)
        self.phase_bins_v = int(phase_bins_v)
        self.phase_v_range = phase_v_range

    def _np_arr(self, a):  # to numpy
        return to_np(a)

    def record_energy(self, it, Ex_xp, v_xp, m, eps0=1.0):
        Ex = self._np_arr(Ex_xp)
        v  = self._np_arr(v_xp)
        dx = self.grid.dx
        WE = 0.5 * _np.sum(Ex**2) * dx * eps0  # total electric energy
        WK = 0.5 * m * _np.mean(_np.sum(v*v, axis=1)) * v.shape[0]  # same scale as WE (sum over particles)
        WT = WE + WK
        self.energy_data.append([it, WE, WK, WT])

        # total amplitude for quick view
        self.total_amps.append(_np.sqrt(_np.sum(Ex**2) * dx))
        return WE, WK, WT

    def record_spectrum(self, it, Ex_xp):
        Ex = self._np_arr(Ex_xp)
        Ek = _np.fft.rfft(Ex)
        Pk = _np.abs(Ek)**2

        # pick kmax index from first meaningful spectrum (ignore DC at k=0)
        if self.kmax_index is None:
            if len(Pk) > 1:
                idx = int(_np.argmax(Pk[1:]) + 1)
            else:
                idx = 0
            self.kmax_index = idx

        # track dominant mode amplitude & phase
        if self.kmax_index is not None and self.kmax_index < len(Ek):
            Ekmax = Ek[self.kmax_index]
            self.kmax_list_steps.append(it)
            self.kmax_amps.append(_np.abs(Ekmax))
            self.kmax_phis.append(_np.angle(Ekmax))

        # save full spectrum frame
        _np.savez(os.path.join(self.outdir, f"spectrum_{it:05d}.npz"),
                  k=self.k_axis, Ek=Ek, Pk=Pk)

    def record_phase_space(self, it, x_xp, vx_xp, grid, 
                       sample_limit=400_000, auto_v=True,
                       v_quantiles=(0.5, 99.5), v_pad=0.15):
        """
        生成 (x, vx) 直方图：
        - auto_v=True: 用分位数自动决定 [vmin, vmax]，避免少数极端值把范围拉太大；
        - 记录 meta=[vmin, vmax, nsample] 便于后处理。
        """
        import numpy as _np
        x  = to_np(x_xp)
        vx = to_np(vx_xp)

        # 子采样：保证有足够样本但不至于太慢
        if x.shape[0] > sample_limit:
            sel = _np.random.choice(x.shape[0], size=sample_limit, replace=False)
            x, vx = x[sel], vx[sel]

        if auto_v:
            qlo, qhi = _np.percentile(vx, v_quantiles)
            width = max(1e-10, qhi - qlo)
            vmin = qlo - v_pad * width
            vmax = qhi + v_pad * width
        else:
            vmin, vmax = self.phase_v_range  # 你原来的固定范围

        H, xedges, vedges = _np.histogram2d(
            x, vx,
            bins=[self.phase_bins_x, self.phase_bins_v],
            range=[[0.0, grid.Lx], [vmin, vmax]],
        )
        _np.savez(os.path.join(self.outdir, f"phase_{it:05d}.npz"),
                H=H, xedges=xedges, vedges=vedges,
                meta=_np.array([vmin, vmax, x.shape[0]], dtype=float))


    def finalize_and_save(self):
        energy_arr = _np.array(self.energy_data, dtype=float)
        _np.savetxt(os.path.join(self.outdir, "energy.txt"), energy_arr,
                    header="step  W_E  W_K  W_T")

        # Estimate growth rates:
        # 1) total growth (from W_E): ln(W_E) ~ 2γ t  => γ = 0.5 * slope
        gamma_total = _np.nan
        if len(energy_arr) >= 4:
            t = energy_arr[:,0]
            WE = energy_arr[:,1]
            # use early ~1/3 as linear window
            n = len(t)//3 if len(t)//3 >= 4 else len(t)
            if n >= 4:
                coeff = _np.polyfit(t[:n], _np.log(_np.maximum(WE[:n], 1e-30)), 1)
                gamma_total = 0.5 * coeff[0]  # in 1/step units

        # 2) dominant mode growth and frequency from tracked Ek(kmax)
        gamma_kmax = _np.nan
        omega_kmax = _np.nan
        kmax_val   = _np.nan

        if self.kmax_index is not None and len(self.kmax_amps) >= 4:
            tk = _np.array(self.kmax_list_steps, dtype=float)
            Ak = _np.array(self.kmax_amps, dtype=float)
            ph = _np.unwrap(_np.array(self.kmax_phis, dtype=float))

            # growth
            n = len(tk)//3 if len(tk)//3 >= 4 else len(tk)
            if n >= 4:
                coeffA = _np.polyfit(tk[:n], _np.log(_np.maximum(Ak[:n], 1e-30)), 1)
                gamma_kmax = coeffA[0]  # since |E_k| ~ e^{γ t} (amplitude)

                # frequency: phase slope = ω_r * dt (if time unit=steps)
                coeffP = _np.polyfit(tk[:n], ph[:n], 1)
                omega_kmax = coeffP[0]  # rad/step

            kmax_val = self.k_axis[self.kmax_index]

        # write summary
        with open(os.path.join(self.outdir, "growth.txt"), "w", encoding="utf-8") as f:
            f.write("# growth/frequency summary (time unit: step)\n")
            f.write(f"kmax_index: {self.kmax_index}\n")
            f.write(f"kmax_value: {kmax_val:.6e}\n")
            f.write(f"gamma_total: {gamma_total:.6e}\n")
            f.write(f"gamma_kmax:  {gamma_kmax:.6e}\n")
            f.write(f"omega_kmax:  {omega_kmax:.6e}\n")

# ---------------------------
# Electric energy helper
# ---------------------------
def electric_energy(Ex_xp):
    Ex = to_np(Ex_xp)
    return 0.5 * float(_np.sum(Ex*Ex))

# ---------------------------
# Main PIC simulator (ES)
# ---------------------------
class PIC1D3V_ES:
    def __init__(self, Lx=2*math.pi, Nx=512, Np=800_000,
                 dt=0.1, steps=1500,
                 v0=1.0, vth=0.1,
                 eps0=1.0, q_e=-1.0, m_e=1.0,
                 outdir="output", diag_interval=50,
                 dtype=xp.float32):
        self.grid   = Grid1D(Lx, Nx, dtype=dtype)
        self.fields = Fields(self.grid, dtype=dtype)
        self.dt     = float(dt)
        self.steps  = int(steps)
        self.eps0   = eps0

        # electrons
        self.elc = Particles(Np, q=q_e, m=m_e, dtype=dtype)
        self._init_positions_uniform()
        self._init_two_stream(v0=v0, vth=vth)

        self.rho = zeros_like_shape(self.grid.Nx, dtype=dtype)

        # diagnostics
        self.diag = Diagnostics(self.grid, outdir=outdir, diag_interval=diag_interval)

    def _init_positions_uniform(self):
        Np = self.elc.Np
        i = xp.arange(Np, dtype=xp.float32)
        rnd = xp.random.random(Np).astype(xp.float32)
        self.elc.x = ((i + rnd) / Np) * self.grid.Lx

    def _init_two_stream(self, v0=1.0, vth=0.1):
        Np = self.elc.Np
        half = Np // 2
        v = self.elc.v
        v[:] = vth * xp.random.standard_normal((Np,3)).astype(v.dtype) # Normal distribution N(0,1) * v_th
        v[:half,0] += v0
        v[half:,0] -= v0

    def step(self):
        grid, elc, f, dt = self.grid, self.elc, self.fields, self.dt

        # 1) Deposit electron charge; neutralize DC (k=0) by subtracting mean
        deposit_charge_CIC(grid, elc, self.rho)
        self.rho -= xp.mean(self.rho) # neutralize 

        # 2) Solve Poisson -> Ex
        f.Ex = solve_poisson_1d(grid, self.rho, eps0=self.eps0)

        # 3) Gather fields (ES: only Ex; EM placeholders kept)
        Ex_p = gather_CIC_Ex(grid, f.Ex, elc.x)
        zeros = xp.zeros_like(Ex_p)
        Ey_p = zeros; Ez_p = zeros
        Bx_p = zeros; By_p = zeros; Bz_p = zeros

        # 4) Boris push
        push_velocity_boris(elc, Ex_p, Ey_p, Ez_p, Bx_p, By_p, Bz_p, dt)

        # 5) Move particles
        push_particle(elc, grid, dt)

        # (EM) 6) Here would be E,B (Yee) updates if EM mode

    def run(self, verbose=True):
        for it in range(self.steps):
            self.step()

            if (it % self.diag.diag_interval) == 0 or (it == self.steps-1):
                # Energy
                WE, WK, WT = self.diag.record_energy(
                    it, self.fields.Ex, self.elc.v, self.elc.m, eps0=self.eps0
                )
                # Spectrum
                self.diag.record_spectrum(it, self.fields.Ex)
                # Phase-space (x,vx)
                self.diag.record_phase_space(it, self.elc.x, self.elc.v[:,0], self.grid)

                if verbose:
                    print(f"[{it:6d}] W_E={WE:.6e}  W_K={WK:.6e}  W_T={WT:.6e}")

        self.diag.finalize_and_save()

# ---------------------------
# Entry
# ---------------------------
if __name__ == "__main__":
    print(f"Backend: {'CuPy (GPU)' if CUPY else 'NumPy (CPU)'}")
    cfg = dict(
        Lx=2*math.pi,
        Nx=512*2,
        Np=800_000,
        dt=0.001,
        steps=2000,
        v0=1.0,
        vth=0.1,
        eps0=1.0, q_e=-1.0, m_e=1.0,
        outdir="output",
        diag_interval=10,
    )
    sim = PIC1D3V_ES(**cfg)
    sim.run(verbose=True)

    print("\nDone. Outputs written to ./output\n"
          " - energy.txt (W_E, W_K, W_T)\n"
          " - growth.txt (gamma_total, gamma_kmax, omega_kmax)\n"
          " - spectrum_*.npz (k, Ek, |Ek|^2)\n"
          " - phase_*.npz (phase-space histograms)\n")
