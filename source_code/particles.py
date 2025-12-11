from .backend import xp, zeros_like_shape
class Particles:
    def __init__(self, Np, Lx, n0, q_sign=-1.0, dtype=xp.float64):
        self.Np = int(Np)
        # Physical form: q_macro = q_sign * e * n0 * Lx / Np, m_macro = m_e * n0 * Lx / Np
        # Normalization: e = m_e = n0 = 1 → q = -Lx/Np, m = Lx/Np
        w = 2 * Lx / Np # n0 is the beam density per stream, normalized to 1
        self.q = dtype(q_sign * w)
        self.m = dtype(w)
        self.x = zeros_like_shape(self.Np, dtype=dtype) # x in [0,Lx)
        self.v = zeros_like_shape((self.Np,3), dtype=dtype) # vx,vy,vz
        self.label = xp.zeros(self.Np, dtype=xp.int8) # different stream labels