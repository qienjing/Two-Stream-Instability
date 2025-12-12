from dataclasses import dataclass, asdict
import math

# ---------------------- Configuration Class ----------------------
@dataclass
class PICConfig:
    # === User input parameters ===
    Lx: float = 30.0          # [λ_D] Domain length
    Nx: int = 32             # Number of grid cells
    Np: int = 500_000         # Number of particles
    dt: float = 0.02        # [1/ω_p] Timestep
    steps: int = 5000
    v0: float = 1.0          # [v_th] Drift velocity
    vth: float = 0.1          # [v_th] Thermal speed
    n0: float = 1.0e15     # [m^-3] beam density per stream
    outdir: str = "output"
    diag_interval: int = 10
    phase_snap: int = 50

    # === Auto-computed quantities ===
    def compute_scales(self):
        """Compute normalization scales (λ_D, ω_p, v_th, etc.)."""
        e_SI = 1.602176634e-19 # C
        m_e_SI = 9.10938356e-31 # kg
        eps0_SI = 8.8541878128e-12 # F/m
        kB_SI = 1.380649e-23 # J/K

        # 1 eV used as a reference
        Te_ref  = 1.0       # eV
        vth_ref_SI = math.sqrt(Te_ref * e_SI / m_e_SI)

        # Normalization reference frequency and length
        omega_p_SI = math.sqrt(self.n0 * e_SI**2 / (eps0_SI * m_e_SI))
        lambdaD_SI = vth_ref_SI / omega_p_SI
        
        print("=== Input Parameters ===")
        self.vth_ref_SI = vth_ref_SI
        self.omega_p_SI = omega_p_SI
        self.lambdaD_SI = lambdaD_SI


        print("=== Normalization (Plasma units) ===")
        print(f"[normalize] v_th={vth_ref_SI:.3e} m/s, ω_p={omega_p_SI:.3e} 1/s, λ_D={lambdaD_SI:.3e} m")
        print(f"n0 = {self.n0:.2e} 10^15 m^-3")
    
        # print("--------------------------------------")
        # print("All quantities below are in normalized (dimensionless) plasma units:")
        # print(" - time scaled by 1/ω_p, length by λ_D, velocity by v_th")
        # print(" - electric field scaled by (m_e v_th ω_p / e)")
        # print(" - charge density scaled by (n0 e)")
        # print(" - energy scaled by (n0 m_e v_th^2)")
        # print("======================================\n")

        # ================================================================
        # In the normalized system, the following constants are set to 1:
        # e = m_e = ε0 = ω_p = λ_D = n0 = v_th (T=1eV) = 1
        # ================================================================
                

    def asdict(self):
        """Convert to a plain dict so **cfg can be passed into classes."""
        return asdict(self)
