import math
from backend import xp

class Grid1D:
    def __init__(self, Lx, Nx, dtype=xp.float64):
        self.Lx = float(Lx) # total length in x
        self.Nx = int(Nx) # number of grid points
        self.dx = self.Lx / self.Nx
        self.dtype = dtype

        # rFFT wave numbers
        # Physical form: k = 2πm / Lx
        # Normalization: Lx normalized by λ_D → k normalized by 1/λ_D
        m = xp.arange(self.Nx//2 + 1, dtype=dtype)
        self.k = (2.0 * math.pi / self.Lx) * m         