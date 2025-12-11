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

Author: Qien Jing, Han Zhao
"""

import os, math, glob
import source_code

# ---------------------------
# Auto-clean output directory
OUT_DIR = "output"
os.makedirs(OUT_DIR, exist_ok=True)
for f in glob.glob(os.path.join(OUT_DIR, "*")):
    if os.path.isfile(f):
        os.remove(f)
    elif os.path.isdir(f):
        import shutil
        shutil.rmtree(f)

# ---------------------------
# Configuration dataclass
from source_code.dataclass import PICConfig

# ---------------------------
# Backend selection (GPU/CPU) + seed
from source_code.backend import xp, fft, to_xp, to_np, zeros_like_shape, roll, scatter_add, CUPY

# ---------------------------
# Grid
from source_code.grid1D import Grid1D
    
# ---------------------------
# Fields container (ES now, EM ready)
from source_code.fields import Fields

# ---------------------------
# Particles (1D position, 3V velocity), Define particle class
from source_code.particles import Particles

# ---------------------------
# Deposition & Gather (CIC)
from source_code.deposit_charge import deposit_charge

# ---------------------------
# Poisson solver (periodic)
from source_code.solve_poisson_1d import solve_poisson_1d

# ---------------------------
# Field gather (CIC)
from source_code.gather_field import gather_field   

# ---------------------------
# Push particles (Boris non-relativistic)
from source_code.push_particle import push_particle

# ---------------------------
# Diagnostics (energy etc.)
from source_code.diagnostics import Diagnostics

# PIC Main class
# ---------------------------
from source_code.PIC1D3V_ES import PIC1D3V_ES
# _init_two_stream, init_push, 
# step(deposit, solve_poisson_1d, gather_field, push_particle), run

# MAIN
if __name__=="__main__":
    print(f"Backend: {'CuPy (GPU)' if CUPY else 'NumPy (CPU)'}")

    # 在 main 里定义输入参数（简洁清晰）
    cfg = PICConfig(
        Lx=30, # domain length in λ_D units (physical Lx = L̂x * λ_D) unit: λ_D
        Nx=64,
        Np=500_000,

        dt=0.02, # normalized dt/ω_pbeam = real Δt
        steps=2000,

        v0=1, # unit: 1eV v_th
        vth=0.8, # unit: 1eV v_th
        n0=1e15, # unit: m^-3, beam density per stream

        diag_interval=10,
        phase_snap=20,
    )

    sim=PIC1D3V_ES(cfg)
    sim.run(verbose=True)

    
