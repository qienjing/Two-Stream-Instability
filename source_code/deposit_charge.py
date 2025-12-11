from .grid1D import Grid1D
from .particles import Particles
from .backend import xp, CUPY, scatter_add
def deposit_charge(grid: Grid1D, parts: Particles, rho_out):
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
