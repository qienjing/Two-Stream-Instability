import os
import numpy as _np
from .backend import to_np, xp, fft
from .grid1D import Grid1D

class Diagnostics:
    def __init__(self, grid: Grid1D, outdir="output", diag_interval=10, phase_snap=50, cfg=None):
        self.grid = grid
        self.outdir = outdir
        self.diag_interval = int(diag_interval)
        self.phase_snap = int(phase_snap)
        self.cfg = cfg
        os.makedirs(outdir, exist_ok=True)

        # Energy Data: [it, W_E, W_K, W_T]
        self.energy_data = []
        # Field Modes Data
        self.modes_data = []

    def record_energy(self, it, Ex_xp, v_xp, m, eps0=1.0):
        Ex = to_np(Ex_xp)
        v  = to_np(v_xp)
        dx = self.grid.dx
        # Physical form: W_E = (ε₀/2) ∑ E² dx , W_K = (m/2) ∑ v²
        # Normalization: ε₀ = 1, Ê = eE/(m_e v_th ω_p), v̂ = v/v_th, dx̂ = dx/λ_D
        # So Ŵ_E = 0.5∑Ê²dx̂ ,  Ŵ_K = 0.5∑v̂²
        WE = 0.5 * _np.sum(Ex**2) * dx
        WK = 0.5 * float(m) * _np.sum(_np.sum(v*v, axis=1))
        WT = WE + WK
        self.energy_data.append([it, WE, WK, WT])
        return WE, WK, WT
    
    def record_phase_space(self, it, x_xp, vx_xp, label_xp, grid, nbins_x=128, nbins_v=128):
        # convert CuPy → NumPy
        x = to_np(x_xp)
        vx = to_np(vx_xp)
        lbl = to_np(label_xp).astype(int)


        # wrap positions to [0, Lx)
        Lx = grid.Lx
        x = _np.mod(x, Lx)

        # choose velocity limits
        vmin, vmax = _np.percentile(vx, [0, 100])  # auto range to ignore outliers
        # === histogram for ALL particles ===
        H_all, xedges, vedges = _np.histogram2d(
        x, vx,
        bins=(nbins_x, nbins_v),
        range=[[0, Lx], [vmin, vmax]]
        )

        # === histogram for label = 0 (+v0 flow) ===
        H0, _, _ = _np.histogram2d(
        x[lbl == 0], vx[lbl == 0],
        bins=(nbins_x, nbins_v),
        range=[[0, Lx], [vmin, vmax]]
        )

        # === histogram for label = 1 (-v0 flow) ===
        H1, _, _ = _np.histogram2d(
        x[lbl == 1], vx[lbl == 1],
        bins=(nbins_x, nbins_v),
        range=[[0, Lx], [vmin, vmax]]
        )

        
        # save everything
        outpath = os.path.join(self.outdir, f"phase_{it:05d}.npz")
        _np.savez(outpath,
              H_all=H_all,
              H0=H0,
              H1=H1,
              xedges=xedges,
              vedges=vedges,
              meta=_np.array([vmin, vmax, nbins_v]))
        outpath = os.path.join(self.outdir, f"phase_{it:05d}.npz")

    def record_fields(self, it, x_grid, rho_xp, phi_xp, Ex_xp):
        """
        Save field distributions: Rho(x), Phi(x), Ex(x).
        """
        # Convert to CPU NumPy
        rho = to_np(rho_xp)
        phi = to_np(phi_xp)
        Ex  = to_np(Ex_xp)

        # Simple x-axis coordinates from 0 to Lx
        x_axis = _np.linspace(0, self.grid.Lx, self.grid.Nx, endpoint=False)

        outpath = os.path.join(self.outdir, f"fields_{it:05d}.npz")
        _np.savez(outpath, x=x_axis, rho=rho, phi=phi, Ex=Ex, step=it)

    # Record k-space modes
    def record_field_modes(self, t, Ex_xp):
        """
        Perform FFT on Ex and record the amplitude of the first few modes.
        t: current simulation time (dt * steps)
        """
        # 1. FFT transform (using cupy.fft or numpy.fft based on backend)
        # Ex_xp is real, so we use rfft
        Ex_k = fft.rfft(Ex_xp)
        
        # 2. Calculate Amplitude |E_k|
        # Normalization: divide by Nx to match physical amplitude definition roughly
        # (Definition varies, but for relative growth rate, consistent scaling is enough)
        Amps = xp.abs(Ex_k) / self.grid.Nx
        
        # 3. Extract first few modes (m=1, 2, 3, 4)
        # m=0 is the DC component (mean field), usually ~0 or irrelevant for instability
        # m=1 is the fundamental mode (k = 2pi/L)
        # Ensure we don't go out of bounds if Nx is very small
        max_m = min(5, len(Amps)) 
        
        # Convert to CPU numpy array for storage
        Amps_np = to_np(Amps)
        
        # Row format: [time, mode_1, mode_2, mode_3, mode_4]
        # We take slice 1:5 which corresponds to indices 1,2,3,4
        modes_of_interest = Amps_np[1:5] 
        
        row = [float(t)] + modes_of_interest.tolist()
        self.modes_data.append(row)

    # Save all basic simulation parameters to params.txt
    def save_params(self):
        cfg = self.cfg
        out = ["### PIC Simulation Parameters\n"]
        for k,v in cfg.__dict__.items():
            out.append(f"{k} = {v}")

        out.append("\n--- Grid ---")
        out.append(f"Lx = {self.grid.Lx}")
        out.append(f"Nx = {self.grid.Nx}")
        out.append(f"dx = {self.grid.dx}")

        with open(os.path.join(self.outdir, "params.txt"), "w") as f:
            f.write("\n".join(out))

    def finalize_and_save(self):
        _np.savetxt(os.path.join(self.outdir,"energy.txt"),
                    _np.array(self.energy_data),
                    header="step  W_E  W_K  W_T")

        _np.savetxt(os.path.join(self.outdir,"modes_history.txt"),
                    _np.array(self.modes_data),
                    header="t  |E1| |E2| |E3| |E4|")

        self.save_params()