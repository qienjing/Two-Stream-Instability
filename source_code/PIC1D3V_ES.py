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
        cfg.compute_scales()   # Compute normalization parameters at initialization

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

    def _init_two_stream(self, SEED=12345):
        ## initalize positions uniformly in [0,Lx)
        # Uniformly distribute macro-particles in [0,Lx)
        Np=self.e.Np
        Lx = self.grid.Lx
        xp.random.seed(SEED)

        # === 1. Uniform positions with random perturbations ===
        i = xp.arange(Np, dtype=xp.float64)
        rnd = xp.random.random(Np).astype(xp.float64)
        x = ((i + rnd) / Np) * Lx

        # === 2. Generate labels: first half +v0, second half -v0 ===
        half = Np // 2 
        labels = xp.concatenate((
            xp.zeros(half, dtype=xp.int8),   # 0 → +v0
            xp.ones(half, dtype=xp.int8)  # 1 → -v0
        ))

        # === 3. Generate velocities with thermal noise ===
        ## initalize velocity for two-stream
        # Physical form: v = ±v0 + thermal noise
        # Normalization: v̂ = v/v_th , hence v0̂ = v0/v_th
        v = self.vth * xp.random.standard_normal((Np, 3)).astype(xp.float64)
        v[:, 0] += self.v0 * (1.0 - 2.0 * labels)   # label=0→+v0, label=1→−v0


        # === 4. Randomly shuffle x ===
        perm = xp.random.permutation(Np)
        self.e.x = x[perm]
        self.e.v = v
        self.e.label = labels

    def init_push(self):
        # v^{0} to v^{−1/2} initialization for Leapfrog
        # 1. For backtracking we need the force at t=0, so do a temporary full field solve.
        self.rho = zeros_like_shape(self.grid.Nx)
        deposit_charge(self.grid, self.e, self.rho)
        Ex_init, _ = solve_poisson_1d(self.grid, self.rho)
        
        # 2. Gather electric field at particle positions
        Ex_p = gather_field(self.grid, Ex_init, self.e.x)

        # 3. Backtrack velocity: v(-1/2) = v(0) - (q/m * E * 0.5*dt)
        q, m, dt = self.e.q, self.e.m, self.dt
        qmdt2 = (q * dt) / (2.0 * m)
        self.e.v[:, 0] -= qmdt2 * Ex_p

    def step(self):
        g,e,f,dt=self.grid,self.e,self.fields,self.dt

        deposit_charge(g,e,self.rho)
        self.rho -= xp.mean(self.rho) # Quasi-neutral
        # self.rho[:] = 0.25 * (xp.roll(self.rho,-1) + 2*self.rho + xp.roll(self.rho,1)) # suppress high-k noise
        f.Ex, f.phi = solve_poisson_1d(g,self.rho)
        Ex_p =gather_field(g,f.Ex,e.x)

        zeros=xp.zeros_like(Ex_p)
        push_particle(e,g,Ex_p,zeros,zeros,zeros,zeros,zeros,dt)

    def run(self,verbose=True):
        self._init_two_stream()
        # initialize_particles(self.e, self.cfg)
        self.init_push()
        for it in range(self.steps):
            WE_prev, WK_prev = 0.0, 0.0   # Initialize before step 0
            self.step()

            # energy diag
            if it > 0 and (it%self.diag.diag_interval)==0 or (it==self.steps-1):
                WE,WK,WT=self.diag.record_energy(it,self.fields.Ex,self.e.v,self.e.m)

                # Record mode amplitudes over time to analyze gamma
                self.diag.record_field_modes(it, self.fields.Ex)
                if verbose:
                    print(f"[{it:6d}] W_E={WE:.6e}  W_K={WK:.6e}  W_T={WT:.6e}")

            # phase space diag
            if it > 0 and (it%self.diag.phase_snap)==0 or (it==self.steps-1):
                self.diag.record_phase_space(it, self.e.x, self.e.v[:,0], self.e.label, self.grid)
                self.diag.record_fields(it, self.grid, self.rho, self.fields.phi, self.fields.Ex)
                # Diagnostic: print CFL number
                vmax = float(xp.max(xp.abs(self.e.v[:,0])))
                dx = self.grid.dx
                cfl = vmax * self.dt / dx
                print(f"[{it:5d}] CFL = {cfl:.3f}  (require < 0.3 for stability)")

                ## Diagnostics
                # (1) Compute current energy
                WE_curr, WK_curr, WT_curr = self.diag.record_energy(
                    it, self.fields.Ex, self.e.v, self.e.m
                )

                # (2) Compute kinetic and electric energy changes
                dWE = WE_curr - WE_prev
                dWK = WK_curr - WK_prev

                # (3) Power closure (energy conservation check)
                g,e,f,dt=self.grid,self.e,self.fields,self.dt
                Ex_p=gather_field(g,f.Ex,e.x)
                P = float(xp.sum(self.e.q * self.e.v[:,0] * Ex_p) * self.dt)
                print(f"[{it:6d}] closure check: dWE+dWK+P = {dWE + dWK + P:+.3e}")

                # (4) Update previous-step energy
                WE_prev, WK_prev = WE_curr, WK_curr

        self.diag.finalize_and_save()