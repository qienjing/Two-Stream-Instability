import math
from .backend import xp
from .grid1D import Grid1D
from .particles import Particles

def push_particle(parts: Particles, grid: Grid1D, Ex_p, Ey_p, Ez_p, Bx_p, By_p, Bz_p, dt):
    """
    Leapfrog Boris pusher (position + velocity, one call).
    Input: x^n and v^{n+1/2} (v_nhalf).
    Use v^{n+1/2} to advance to x^{n+1}, then apply Boris to obtain v^{n+3/2}.

    """
    # Physical form: dv/dt = (q/m)(E + v×B)
    # Normalization: q/m = -1, Ê = eE/(m_e v_th ω_p), t̂ = ω_p t
    # Physical form: x ← x + v_x dt
    # Normalization: x̂ = x / λ_D, v̂ = v / v_th, dt̂ = ω_p dt
    q, m = parts.q, parts.m
    v = parts.v
    x = parts.x

    # (1) Boris: v^{n-1/2} + E^{n} -> v^{n+1/2}
    qmdt2 = (q * dt) / (2.0 * m)

    # First half-step E-kick: u = v^{n-1/2} + (qE/m) dt/2
    ux = v[:,0] + qmdt2 * Ex_p
    uy = v[:,1] + qmdt2 * Ey_p
    uz = v[:,2] + qmdt2 * Ez_p

    # B rotation (supports adding magnetic field later; electrostatic sets Bp=0)
    hx = qmdt2 * Bx_p
    hy = qmdt2 * By_p
    hz = qmdt2 * Bz_p
    h2 = hx*hx + hy*hy + hz*hz
    sx = 2.0 * hx / (1.0 + h2)
    sy = 2.0 * hy / (1.0 + h2)
    sz = 2.0 * hz / (1.0 + h2)

    vpx = ux + (uy * hz - uz * hy)
    vpy = uy + (uz * hx - ux * hz)
    vpz = uz + (ux * hy - uy * hx)

    upx = ux + (vpy * sz - vpz * sy)
    upy = uy + (vpz * sx - vpx * sz)
    upz = uz + (vpx * sy - vpy * sx)

    # Second half-step E-kick: v^{n+1/2} = v^+ = v' + (qE/m) dt/2, preparing for next step
    v[:,0] = upx + qmdt2 * Ex_p
    v[:,1] = upy + qmdt2 * Ey_p
    v[:,2] = upz + qmdt2 * Ez_p

    # (2) x^{n+1} = x^n + v^{n+1/2} dt
    x += v[:,0] * dt
    x[:] = xp.mod(x, grid.Lx)   # periodic BC