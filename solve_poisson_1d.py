from grid1D import Grid1D
from backend import xp, fft

def solve_poisson_1d(grid: Grid1D, rho, eps0=1.0):
    # Physical form: ∂²φ/∂x² = -ρ/ε₀ → φ_k = -ρ_k / (ε₀ k²)
    # Normalization: ε₀ = 1, ρ̂ = ρ / (n0 e), φ̂ = eφ / (m_e v_th²)
    Nx = grid.Nx
    k = grid.k
    rho_k = fft.rfft(rho)
    phi_k = xp.zeros_like(rho_k, dtype=rho_k.dtype)
    phi_k[0] = 0.0                     # ✅ 去除 k=0 模 (DC component)
    
    phi_k[1:] = rho_k[1:] / (eps0 * (k[1:]**2))
    phi = fft.irfft(phi_k, n=Nx).astype(rho.dtype)
    Ex_k = -1j * k * phi_k  # E_k = -ik φ_k
    Ex = fft.irfft(Ex_k, n=Nx).astype(rho.dtype)
    return Ex, phi