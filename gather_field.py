import math
from backend import xp
from grid1D import Grid1D

def gather_field(grid: Grid1D, fld, x_particles):
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