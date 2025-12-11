from dataclasses import dataclass, asdict
import math
# ---------------------- Configuration Class ----------------------
@dataclass
class PICConfig:
    # === 用户输入参数 ===
    Lx: float = 30.0          # [λ_D] 盒长
    Nx: int = 32             # 网格数
    Np: int = 500_000         # 粒子数
    dt: float = 0.02        # [1/ω_p] 步长
    steps: int = 5000
    v0: float = 1.0          # [v_th] 漂移速度
    vth: float = 0.1          # [v_th] 热速
    n0: float = 1.0e15     # [m^-3] beam density per stream
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

        # 1 eV 作为标准参考（可以自由改成你的基准）
        Te_ref  = 1.0       # eV
        vth_ref_SI = math.sqrt(Te_ref * e_SI / m_e_SI)

        # 归一化参考频率和长度
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
        # 在归一化体系中，以下常数均设为1：
        # e = m_e = ε0 = ω_p = λ_D = n0 = v_th (T=1eV) = 1
        # ================================================================
                

    def asdict(self):
        """转换为普通 dict，便于 **cfg 传入类"""
        return asdict(self)