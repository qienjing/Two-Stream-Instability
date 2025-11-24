from dataclasses import dataclass, asdict
import math
# ---------------------- Configuration Class ----------------------
@dataclass
class PICConfig:
    # === 用户输入参数 ===
    Lx: float = 30.0          # [λ_D] 盒长
    Nx: int = 256             # 网格数
    Np: int = 800_000         # 粒子数
    dt: float = 0.005         # [1/ω_p] 步长
    steps: int = 5000
    v0: float = 5.0           # [v_th] 漂移速度
    vth: float = 1.0          # [v_th] 热速
    n0: float = 1.0e15     # [m^-3]
    Te: float = 1.0        # [eV]
    outdir: str = "output"
    diag_interval: int = 10
    phase_snap: int = 50

    # === 自动计算量 ===
    def compute_scales(self):
        """计算归一化尺度（λ_D, ω_p, v_th 等）"""
        e_SI = 1.602176634e-19 # C
        m_e_SI = 9.10938356e-31 # kg
        eps0_SI = 8.8541878128e-12 # F/m
        kB_SI = 1.380649e-23 # J/K
        vth = math.sqrt(self.Te * e_SI / m_e_SI)  # 热速，单位 m/s

        # 以 1e15 m^-3 和 1 eV 作为标准参考（可以自由改成你的基准）
        n0_ref  = 1.0e15    # m^-3, total density
        Te_ref  = 1.0       # eV
        self.n0 = self.n0 / n0_ref
        self.Te = self.Te / Te_ref

        # use ref parameter to calculate normalization scales
        vth_ref_SI = math.sqrt(Te_ref * e_SI / m_e_SI)
        omega_p_SI = math.sqrt(n0_ref * e_SI**2 / (eps0_SI * m_e_SI))
        lambdaD_SI = vth_ref_SI / omega_p_SI

        self.vth = vth / vth_ref_SI
        print("=== Input Parameters ===")
        print(self.vth, self.v0)
        
        self.vth_ref_SI = vth_ref_SI
        self.omega_p_SI = omega_p_SI
        self.lambdaD_SI = lambdaD_SI


        print("=== Normalization (Plasma units) ===")
        print(f"[normalize] v_th={vth_ref_SI:.3e} m/s, ω_p={omega_p_SI:.3e} 1/s, λ_D={lambdaD_SI:.3e} m")
        print(f"T_e = {self.Te:.2f} eV, n0 = {self.n0:.2e} 10^15 m^-3")
        print(f"To recover SI: multiply by n0*m_e*v_th^2 = {self.n0*n0_ref*m_e_SI*vth_ref_SI**2:.3e} J per unit area")
        # print("--------------------------------------")
        # print("All quantities below are in normalized (dimensionless) plasma units:")
        # print(" - time scaled by 1/ω_p, length by λ_D, velocity by v_th")
        # print(" - electric field scaled by (m_e v_th ω_p / e)")
        # print(" - charge density scaled by (n0 e)")
        # print(" - energy scaled by (n0 m_e v_th^2)")
        # print("======================================\n")

        # ================================================================
        # 在归一化体系中，以下常数均设为1：
        # e = m_e = ε0 = ω_p = λ_D = v_th = 1
        # ================================================================
                

    def asdict(self):
        """转换为普通 dict，便于 **cfg 传入类"""
        return asdict(self)