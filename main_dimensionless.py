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
from dataclass import PICConfig


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

def zeros_like_shape(shape, dtype=xp.float64):
    return xp.zeros(shape, dtype=dtype)

def roll(a, shift, axis):
    return xp.roll(a, shift, axis=axis)

# ---------------------------
# Grid
# ---------------------------
class Grid1D:
    def __init__(self, Lx, Nx, dtype=xp.float64):
        self.Lx = float(Lx) # total length in x
        self.Nx = int(Nx) # number of grid points
        self.dx = self.Lx / self.Nx
        self.dtype = dtype

        # rFFT wave numbers
        # Physical form: k = 2πm / Lx
        # Normalization: Lx normalized by λ_D → k normalized by 1/λ_D
        m = xp.arange(self.Nx//2 + 1, dtype=dtype)
        k_cont = (2.0 * math.pi / self.Lx) * m          # 连续 k
        self.k = k_cont
        # 离散拉普拉斯本征值用 k_tilde = (2/dx) sin(k dx / 2)
        kd = (2.0 / self.dx) * xp.sin(0.5 * k_cont * self.dx)
        self.kd = kd # shape (Nx/2+1,)

        self.k = (2.0 * math.pi / self.Lx) * m # shape (Nx/2+1,)

# ---------------------------
# Fields container (ES now, EM ready)
# ---------------------------
class Fields:
    def __init__(self, grid: Grid1D, dtype=xp.float64):
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
    def __init__(self, Np, Lx, q_sign=-1.0, dtype=xp.float64):
        self.Np = int(Np)
        # Physical form: q_macro = q_sign * e * n0 * Lx / Np, m_macro = m_e * n0 * Lx / Np
        # Normalization: e = m_e = n0 = 1 → q = -Lx/Np, m = Lx/Np
        w = Lx / Np
        self.q = dtype(q_sign * w)
        self.m = dtype(w)
        self.x = zeros_like_shape(self.Np, dtype=dtype) # x in [0,Lx)
        self.v = zeros_like_shape((self.Np,3), dtype=dtype) # vx,vy,vz
        self.label = xp.zeros(self.Np, dtype=xp.int8) # different stream labels

# ---------------------------
# Deposition & Gather (CIC)
# ---------------------------
## Deposit charge density from particles to grid
def deposit_charge_CIC(grid: Grid1D, parts: Particles, rho_out):
    Nx, dx = grid.Nx, grid.dx
    x = parts.x
    q = parts.q

    # Physical form: ρ(x) = Σ q S(x - x_p)
    # Normalization: ρ̂ = ρ / (n0 e)
    xc = x / dx # grid index
    iL = xp.floor(xc).astype(xp.int32) # left grid point index
    x_incell = xc - iL # position in cell [0,1)
    iR = (iL + 1) % Nx

    wL = 1.0 - x_incell # weight of a particle on left grid point
    wR = x_incell
    # charge per length per particle contribution
    contrib_L = q * wL / dx
    contrib_R = q * wR / dx

    rho_out.fill(0.0) # reset
    if CUPY:
        scatter_add(rho_out, iL, contrib_L)
        scatter_add(rho_out, iR, contrib_R)
    else:
        xp.add.at(rho_out, iL, contrib_L)
        xp.add.at(rho_out, iR, contrib_R)

# ---------------------------
# Field gather (CIC)
# ---------------------------
def gather_CIC_field(grid: Grid1D, fld, x_particles):
    """
    CIC gather for a scalar grid field 'fld' (Ex, Ey, Ez, Bx, By, Bz).
    Physical form: F_p = Σ_i F_i S(x_i - x_p)
    """
    # Physical form: E_p = Σ E_i S(x_i - x_p)
    # Normalization: Ê = eE / (m_e v_th ω_p)
    Nx, dx = grid.Nx, grid.dx
    xc = x_particles / dx
    iL = xp.floor(xc).astype(xp.int32)
    x_incell = xc - iL
    iR = (iL + 1) % Nx
    wL = 1.0 - x_incell
    wR = x_incell
    return wL * fld[iL] + wR * fld[iR]


# ---------------------------
# Poisson solver (periodic)
# ---------------------------
def solve_poisson_1d(grid: Grid1D, rho, eps0=1.0):
    # Physical form: ∂²φ/∂x² = -ρ/ε₀ → φ_k = -ρ_k / (ε₀ k²)
    # Normalization: ε₀ = 1, ρ̂ = ρ / (n0 e), φ̂ = eφ / (m_e v_th²)
    Nx = grid.Nx
    k = grid.k
    rho_k = fft.rfft(rho)
    phi_k = xp.zeros_like(rho_k, dtype=rho_k.dtype)

    phi_k[0] = 0.0                     # ✅ 去除 k=0 模 (DC component)
    kd = grid.kd
    # phi_k[1:] = -rho_k[1:] / (eps0 * (k[1:]**2))
    # 用离散波数，避免 k^2 与网格拉普拉斯不配
    phi_k[1:] = -rho_k[1:] / (eps0 * (kd[1:]**2))

    # E_k = -ik φ_k → Ê = eE / (m_e v_th ω_p)
    # Ex_k = -1j * k * phi_k # E = -∂x φ → 仍用连续 k 做导数是可以的
    # ✅ 用“离散梯度”而不是连续 k
    kg = (1.0 / grid.dx) * xp.sin(grid.k * grid.dx)  # Γ = sin(k dx)/dx
    Ex_k = -1j * kg * phi_k
    Ex = fft.irfft(Ex_k, n=Nx).astype(rho.dtype)
    return Ex

# ---------------------------
# Push particles (Boris non-relativistic)
# ---------------------------
def half_step_preheat(parts: Particles, Ex_p, Ey_p, Ez_p, dt):
    """
    Half-step preheating (E/2 kick).
    把速度从整步 v^0 推到半步 v^{1/2} = v_nhalf。
    """
    q, m = parts.q, parts.m
    coef = 0.5 * dt * (q / m)
    parts.v[:,0] += coef * Ex_p
    parts.v[:,1] += coef * Ey_p
    parts.v[:,2] += coef * Ez_p


def push_particle(parts: Particles, grid: Grid1D, Ex_p, Ey_p, Ez_p, Bx_p, By_p, Bz_p, dt):
    """
    Leapfrog Boris pusher (position + velocity, one call).
    输入：x^n 与 v^{n+1/2} (v_nhalf)，
    用 v^{n+1/2} 推到 x^{n+1}；再用 Boris 得到 v^{n+3/2}。

    时间层说明：
      - 入参 v 为 v_nhalf (半步速度)
      - 更新后 v 变为 v_{(n+1)half} = v^{n+3/2}
    """
    # Physical form: dv/dt = (q/m)(E + v×B)
    # Normalization: q/m = -1, Ê = eE/(m_e v_th ω_p), t̂ = ω_p t
    # Physical form: x ← x + v_x dt
    # Normalization: x̂ = x / λ_D, v̂ = v / v_th, dt̂ = ω_p dt
    q, m = parts.q, parts.m
    v = parts.v
    x = parts.x

    # (a) x^{n+1} = x^n + v^{n+1/2} dt
    x += v[:,0] * dt
    x[:] = xp.mod(x, grid.Lx)   # periodic BC

    # (b) Boris: v^{n+1/2} -> v^{n+3/2}
    qmdt2 = (q * dt) / (2.0 * m)

    # 预半步 E-kick：v^- = v^{n+1/2} + (qE/m) dt/2
    vxm = v[:,0] + qmdt2 * Ex_p
    vym = v[:,1] + qmdt2 * Ey_p
    vzm = v[:,2] + qmdt2 * Ez_p

    # B 旋转（支持未来加入磁场；电静态时 Bp=0 即可）
    hx = qmdt2 * Bx_p
    hy = qmdt2 * By_p
    hz = qmdt2 * Bz_p
    h2 = hx*hx + hy*hy + hz*hz
    sx = 2.0 * hx / (1.0 + h2)
    sy = 2.0 * hy / (1.0 + h2)
    sz = 2.0 * hz / (1.0 + h2)

    vpx = vxm + (vym*sz - vzm*sy)
    vpy = vym + (vzm*sx - vxm*sz)
    vpz = vzm + (vxm*sy - vym*sx)

    # 后半步 E-kick：v^{n+3/2} = v^+ = v' + (qE/m) dt/2, 这一步是为了下一次推进做准备
    v[:,0] = vpx + qmdt2 * Ex_p
    v[:,1] = vpy + qmdt2 * Ey_p
    v[:,2] = vpz + qmdt2 * Ez_p


# ---------------------------
# Diagnostics (energy etc.)
# ---------------------------
import numpy as _np
class Diagnostics:
    def __init__(self, grid: Grid1D, outdir="output", diag_interval=10, phase_snap=50):
        self.grid = grid
        self.outdir = outdir
        self.diag_interval = int(diag_interval)
        self.phase_snap = int(phase_snap)
        os.makedirs(outdir, exist_ok=True)
        self.energy_data = []
        self.k_axis = (2*_np.pi/grid.Lx) * _np.arange(grid.Nx//2 + 1)
        self.kmax_index = None
        self.kmax_list_steps = []
        self.kmax_amps = []
        self.kmax_phis = []
        self.total_amps = []

    def record_energy(self, it, Ex_xp, v_xp, m, eps0=1.0):
        Ex = to_np(Ex_xp)
        v  = to_np(v_xp)
        dx = self.grid.dx
        # Physical form: W_E = (ε₀/2) ∑ E² dx , W_K = (m/2) ∑ v²
        # Normalization: ε₀ = 1, Ê = eE/(m_e v_th ω_p), v̂ = v/v_th, dx̂ = dx/λ_D
        # So Ŵ_E = 0.5∑Ê²dx̂ ,  Ŵ_K = 0.5∑v̂²
        WE = 0.5 * _np.sum(Ex**2) * dx
        WK = 0.5 * float(m) * _np.sum(_np.sum(v*v, axis=1))
        WT = WE + WK
        self.energy_data.append([it, WE, WK, WT])
        return WE, WK, WT
    
    def record_phase_space(self, it, x_xp, vx_xp, grid, nbins_x=128, nbins_v=128):
        # convert CuPy → NumPy
        x = to_np(x_xp)
        vx = to_np(vx_xp)

        # wrap positions to [0, Lx)
        Lx = grid.Lx
        x = _np.mod(x, Lx)

        # choose velocity limits
        vmin, vmax = _np.percentile(vx, [1, 99])  # auto range to ignore outliers

        # make histogram
        H, xedges, vedges = _np.histogram2d(x, vx, bins=(nbins_x, nbins_v),
                                           range=[[0, Lx], [vmin, vmax]])

        # save
        outpath = os.path.join(self.outdir, f"phase_{it:05d}.npz")
        _np.savez(outpath, H=H, xedges=xedges, vedges=vedges,
                 meta=_np.array([vmin, vmax, nbins_v]))


    def finalize_and_save(self):
        energy_arr = _np.array(self.energy_data,float)
        _np.savetxt(os.path.join(self.outdir,"energy.txt"),energy_arr,
                    header="step  W_E_hat  W_K_hat  W_T_hat  (dimensionless)")
    

# ---------------------------
# PIC Main class
# ---------------------------
class PIC1D3V_ES:
    def __init__(self, cfg: PICConfig):

        self.cfg = cfg
        cfg.compute_scales()   # 初始化时计算归一化参数

        self.grid = Grid1D(cfg.Lx,cfg.Nx)
        self.fields=Fields(self.grid)
        self.dt=float(cfg.dt)
        self.steps=int(cfg.steps)
        self.e=Particles(cfg.Np,cfg.Lx,q_sign=-1.0)
        self.v0=float(cfg.v0)
        self.vth=float(cfg.vth)
        self._init_two_stream()
        self.rho=zeros_like_shape(self.grid.Nx)
        self.diag=Diagnostics(self.grid,cfg.outdir,cfg.diag_interval,cfg.phase_snap)

    def _init_two_stream(self):
        ## initalize positions uniformly in [0,Lx)
        # Uniformly distribute macro-particles in [0,Lx)
        Np=self.e.Np
        Lx = self.grid.Lx

        # === 1. 均匀分布位置 + 随机扰动 ===
        i = xp.arange(Np, dtype=xp.float64)
        rnd = xp.random.random(Np).astype(xp.float64)
        x = ((i + rnd) / Np) * Lx

        # === 2. 生成标签：前半 +v0，后半 -v0 ===
        half = Np // 2
        labels = xp.concatenate((
            xp.zeros(half, dtype=xp.int8),   # 0 → +v0
            xp.ones(Np - half, dtype=xp.int8)  # 1 → -v0
        ))

        # === 3. 生成速度（带热噪声） ===
        ## initalize velocity for two-stream
        # Physical form: v = ±v0 + thermal noise
        # Normalization: v̂ = v/v_th , hence v0̂ = v0/v_th
        v = self.vth * xp.random.standard_normal((Np, 3)).astype(xp.float64)
        v[:, 0] += self.v0 * (1.0 - 2.0 * labels)   # label=0→+v0, label=1→−v0


        # === 4. 统一随机打乱（x、v、label 都一起 perm） ===
        # by only disrupting the order of positions, ensure uniform spatial distribution 
        perm = xp.random.permutation(Np)
        self.e.x = x[perm]
        self.e.v = v
        self.e.label = labels


    def step(self, first_step=False):
        g,e,f,dt=self.grid,self.e,self.fields,self.dt
        deposit_charge_CIC(g,e,self.rho)
        # Neutralize background (remove DC component)
        # Physical form: add +n0e to balance mean(ρ)
        self.rho -= xp.mean(self.rho) # Quasi-neutral
        # self.rho[:] = 0.25 * (xp.roll(self.rho,-1) + 2*self.rho + xp.roll(self.rho,1)) # 抑制高k噪声
        f.Ex = solve_poisson_1d(g,self.rho)
        Ex_p=gather_CIC_field(g,f.Ex,e.x)
        zeros=xp.zeros_like(Ex_p)

        if first_step:
            half_step_preheat(e,Ex_p,zeros,zeros,dt)
        else:
            push_particle(e,g,Ex_p,zeros,zeros,zeros,zeros,zeros,dt)

    def run(self,verbose=True):
        # ---- 初始化后，做一次半步预热 ----
        self.step(first_step=True)
        for it in range(self.steps):
            WE_prev, WK_prev = 0.0, 0.0   # 初始化（第0步之前）
            self.step()

            # energy diag
            if (it%self.diag.diag_interval)==0 or (it==self.steps-1):
                WE,WK,WT=self.diag.record_energy(it,self.fields.Ex,self.e.v,self.e.m)
                if verbose:
                    print(f"[{it:6d}] W_E={WE:.6e}  W_K={WK:.6e}  W_T={WT:.6e}")

            # phase space diag
            if (it%self.diag.phase_snap)==0 or (it==self.steps-1):
                self.diag.record_phase_space(it, self.e.x, self.e.v[:,0], self.grid)

                # 诊断 打印 CFL 数
                vmax = float(xp.max(xp.abs(self.e.v[:,0])))
                dx = self.grid.dx
                cfl = vmax * self.dt / dx
                print(f"[{it:5d}] CFL = {cfl:.3f}  (require < 0.3 for stability)")

                ## 诊断
                # （1）计算当前能量
                WE_curr, WK_curr, WT_curr = self.diag.record_energy(
                    it, self.fields.Ex, self.e.v, self.e.m
                )

                # （2）计算动能/电场能变化量
                dWE = WE_curr - WE_prev
                dWK = WK_curr - WK_prev

                # （3）功率闭合（能量守恒检查）
                g,e,f,dt=self.grid,self.e,self.fields,self.dt
                Ex_p=gather_CIC_field(g,f.Ex,e.x)
                P = float(xp.sum(self.e.q * self.e.v[:,0] * Ex_p) * self.dt)
                print(f"[{it:6d}] closure check: dWE+dWK+P = {dWE + dWK + P:+.3e}")

                # （4）更新“前一步”的能量
                WE_prev, WK_prev = WE_curr, WK_curr

        self.diag.finalize_and_save()


# ---------------------------
# Entry point
if __name__=="__main__":
    print(f"Backend: {'CuPy (GPU)' if CUPY else 'NumPy (CPU)'}")

    # 在 main 里定义输入参数（简洁清晰）
    cfg = PICConfig(
        Lx=30.0, # domain length in λ_D units (physical Lx = L̂x * λ_D) unit: λ_D
        Nx=512,
        Np=800_000,

        dt=0.002, # normalized dt = ω_p * Δt
        steps=1000,

        v0=2.0, # unit: v_th
        n0=1e15, # unit: m^-3
        Te=5.0, # unit: eV

        diag_interval=10,
        phase_snap=50,
    )

    sim=PIC1D3V_ES(cfg)
    sim.run(verbose=True)

    
