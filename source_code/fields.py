from .backend import xp, zeros_like_shape
from .grid1D import Grid1D
class Fields:
    def __init__(self, grid: Grid1D, dtype=xp.float64):
        Nx = grid.Nx
        self.phi = zeros_like_shape(Nx, dtype=dtype)
        self.Ex = zeros_like_shape(Nx, dtype=dtype)
        self.Ey = zeros_like_shape(Nx, dtype=dtype)
        self.Ez = zeros_like_shape(Nx, dtype=dtype)
        self.Bx = zeros_like_shape(Nx, dtype=dtype)
        self.By = zeros_like_shape(Nx, dtype=dtype)
        self.Bz = zeros_like_shape(Nx, dtype=dtype)