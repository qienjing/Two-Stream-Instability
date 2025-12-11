# test_gather_field.py
import numpy as np
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import source_code
from source_code.backend import xp
from source_code.grid1D import Grid1D
from source_code.gather_field import gather_field   
from source_code.fields import Fields              
from source_code.particles import Particles

"""
Test objective:
Verify that gather_field correctly performs CIC interpolation:
    F_p = wL * F[iL] + wR * F[iR]

Test method:
1. Construct a known linear field Ex(x) = x (or a sinusoidal field)
2. Manually place particles at several specified positions
3. Compute the analytical values and compare them with gather_field results
"""

def test_gather_field_linear():
    # ---- Construct a simple grid ----
    Lx = 10.0
    Nx = 10
    grid = Grid1D(Lx, Nx)
    dx = grid.dx

    # ---- Construct a linear field: Ex[i] = i ----
    # This allows analytical results to be computed easily
    Ex = xp.arange(Nx, dtype=xp.float64)

    # ---- Manually place several particles ----
    x_particles = xp.array([
        0.1 * dx,      # Near cell 0
        3.7 * dx,      # Located between cell 3 and 4
        9.3 * dx       # Near the boundary wrap-around
    ], dtype=xp.float64)

    # ---- Call gather ----
    Ex_p = gather_field(grid, Ex, x_particles)

    # ---- Analytical verification ----
    # p0: position = 0.1 dx -> xc=0.1
    #  iL=0, iR=1, wL=0.9, wR=0.1
    #  Ex_p = 0.9*0 + 0.1*1 = 0.1

    # p1: position = 3.7 dx -> xc=3.7
    #  iL=3, iR=4, wL=0.3, wR=0.7
    #  Ex_p = 0.3*3 + 0.7*4 = 3.7
