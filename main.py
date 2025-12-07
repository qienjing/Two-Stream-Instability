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

# ---------------------------
# Configuration dataclass
from dataclass import PICConfig

# ---------------------------
# Backend selection (GPU/CPU) + seed
from backend import xp, fft, to_xp, to_np, zeros_like_shape, roll, scatter_add, CUPY

# ---------------------------
# Grid
from grid1D import Grid1D
    
# ---------------------------
# Fields container (ES now, EM ready)
from fields import Fields

# ---------------------------
# Particles (1D position, 3V velocity), Define particle class
from particles import Particles

# ---------------------------
# Deposition & Gather (CIC)
from deposit_charge import deposit_charge

# ---------------------------
# Poisson solver (periodic)
from solve_poisson_1d import solve_poisson_1d

# ---------------------------
# Field gather (CIC)
from gather_field import gather_field   

# ---------------------------
# Push particles (Boris non-relativistic)
from push_particle import push_particle

# ---------------------------
# Diagnostics (energy etc.)
from diagnostics import Diagnostics

# PIC Main class
# ---------------------------
from PIC1D3V_ES import PIC1D3V_ES
# _init_two_stream, init_push, 
# step(deposit, solve_poisson_1d, gather_field, push_particle), run

# MAIN
if __name__=="__main__":
    print(f"Backend: {'CuPy (GPU)' if CUPY else 'NumPy (CPU)'}")

    # 在 main 里定义输入参数（简洁清晰）
    cfg = PICConfig(
        Lx=30, # domain length in λ_D units (physical Lx = L̂x * λ_D) unit: λ_D
        Nx=64,
        Np=100_000,

        dt=0.02, # normalized dt/ω_pbeam = real Δt
        steps=1000,

        v0=1, # unit: 1eV v_th
        vth=0.1, # unit: 1eV v_th
        n0=1e15, # unit: m^-3, beam density per stream

        diag_interval=10,
        phase_snap=20,
    )

    sim=PIC1D3V_ES(cfg)
    sim.run(verbose=True)

    
