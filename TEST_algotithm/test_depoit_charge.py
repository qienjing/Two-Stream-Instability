import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import source_code
from source_code.backend import xp, CUPY, scatter_add
from source_code.grid1D import Grid1D
from source_code.particles import Particles
from source_code.deposit_charge import deposit_charge


# ---------------------
# 1. grid setup
# ---------------------
Nx = 20
Lx = 1.0
grid = Grid1D(Nx=Nx, Lx=Lx)
dx = grid.dx

# ---------------------
# 2. define 3 particles
# ---------------------
parts = type("P", (), {})()

parts.x = xp.array([
    0.10,        # particle near left boundary
    0.37,        # arbitrary point
    0.96         # near right boundary → tests periodic wrapping
])

parts.q = xp.array([
    -1.0,        # negative charge
     0.5,        # positive charge
    -0.2         # small negative charge
])

rho = xp.zeros(Nx)

# ---------------------
# 3. Call your deposition function
# ---------------------
deposit_charge(grid, parts, rho)

# ---------------------
# 4. Print result
# ---------------------
print("Grid charge density ρ:")
print(xp.asarray(rho))

# ---------------------
# 5. Check global charge conservation
# ---------------------
total_rho = float(xp.sum(rho) * dx)
total_q   = float(xp.sum(parts.q))

print("\nTotal deposited charge =", total_rho)
print("Sum of particle charges =", total_q)
print("Charge error =", abs(total_rho - total_q))