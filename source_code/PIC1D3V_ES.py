import math
from .backend import xp, to_np, zeros_like_shape
from .grid1D import Grid1D
from .particles import Particles
from .fields import Fields
from .initial import initialize_particles
from .solve_poisson_1d import solve_poisson_1d
from .deposit_charge import deposit_charge
from .gather_field import gather_field
from .push_particle import push_particle
from .diagnostics import Diagnostics
from .dataclass import PICConfig

class PIC1D3V_ES:
    def __init__(self, cfg: PICConfig):

        self.cfg = cfg
        cfg.compute_scales()   # 初始化时计算归一化参数

        self.grid = Grid1D(cfg.Lx,cfg.Nx)
        self.fields=Fields(self.grid)
        self.dt=float(cfg.dt)
        self.steps=int(cfg.steps)
        self.e=Particles(cfg.Np,cfg.Lx,cfg.n0,q_sign=-1.0)
        self.v0=float(cfg.v0)
        self.vth=float(cfg.vth)
        self._init_two_stream()
        self.rho=zeros_like_shape(self.grid.Nx)
        self.diag=Diagnostics(self.grid,cfg.outdir,cfg.diag_interval,cfg.phase_snap, cfg=cfg)

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
            xp.ones(half, dtype=xp.int8)  # 1 → -v0
        ))

        # === 3. 生成速度（带热噪声） ===
        ## initalize velocity for two-stream
        # Physical form: v = ±v0 + thermal noise
        # Normalization: v̂ = v/v_th , hence v0̂ = v0/v_th
        v = self.vth * xp.random.standard_normal((Np, 3)).astype(xp.float64)
        v[:, 0] += self.v0 * (1.0 - 2.0 * labels)   # label=0→+v0, label=1→−v0


        # === 4. 随机打乱 x ===
        perm = xp.random.permutation(Np)
        self.e.x = x[perm]
        self.e.v = v
        self.e.label = labels

    def init_push(self):
        # v^{0} to v^{−1/2} initialization for Leapfrog
        # 1. 为了回溯，我们需要 t=0 时刻的受力，因此需要临时做一次完整的场求解。  
        self.rho = zeros_like_shape(self.grid.Nx)
        deposit_charge(self.grid, self.e, self.rho)
        Ex_init, _ = solve_poisson_1d(self.grid, self.rho)
        
        # 2. 得到粒子位置处的电场力
        Ex_p = gather_field(self.grid, Ex_init, self.e.x)

        # 3. 回溯速度： v(-1/2) = v(0) - (q/m * E * 0.5*dt)
        q, m, dt = self.e.q, self.e.m, self.dt
        qmdt2 = (q * dt) / (2.0 * m)
        self.e.v[:, 0] -= qmdt2 * Ex_p

    def step(self):
        g,e,f,dt=self.grid,self.e,self.fields,self.dt

        deposit_charge(g,e,self.rho)
        self.rho -= xp.mean(self.rho) # Quasi-neutral
        # self.rho[:] = 0.25 * (xp.roll(self.rho,-1) + 2*self.rho + xp.roll(self.rho,1)) # 抑制高k噪声
        f.Ex, f.phi = solve_poisson_1d(g,self.rho)
        Ex_p =gather_field(g,f.Ex,e.x)

        zeros=xp.zeros_like(Ex_p)
        push_particle(e,g,Ex_p,zeros,zeros,zeros,zeros,zeros,dt)

    def run(self,verbose=True):
        # self._init_two_stream()
        initialize_particles(self.e, self.cfg)
        self.init_push()
        for it in range(self.steps):
            WE_prev, WK_prev = 0.0, 0.0   # 初始化（第0步之前）
            self.step()

            # energy diag
            if it > 0 and (it%self.diag.diag_interval)==0 or (it==self.steps-1):
                WE,WK,WT=self.diag.record_energy(it,self.fields.Ex,self.e.v,self.e.m)

                # 记录模数随时间的变化，用于分析 gamma
                self.diag.record_field_modes(it, self.fields.Ex)
                if verbose:
                    print(f"[{it:6d}] W_E={WE:.6e}  W_K={WK:.6e}  W_T={WT:.6e}")

            # phase space diag
            if it > 0 and (it%self.diag.phase_snap)==0 or (it==self.steps-1):
                self.diag.record_phase_space(it, self.e.x, self.e.v[:,0], self.e.label, self.grid)
                self.diag.record_fields(it, self.grid, self.rho, self.fields.phi, self.fields.Ex)
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
                Ex_p=gather_field(g,f.Ex,e.x)
                P = float(xp.sum(self.e.q * self.e.v[:,0] * Ex_p) * self.dt)
                print(f"[{it:6d}] closure check: dWE+dWK+P = {dWE + dWK + P:+.3e}")

                # （4）更新“前一步”的能量
                WE_prev, WK_prev = WE_curr, WK_curr

        self.diag.finalize_and_save()