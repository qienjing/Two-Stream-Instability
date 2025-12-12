# Two Stream Instability

## Project Description
This repository implements a 1D3V (one dimensional, three velocity component) electrostatic Particle In Cell (PIC) simulator for studying two stream instability. The modular design keeps the core steps: charge deposition, Poisson solving, field gathering, particle pushing and diagnostics. The code can be run with either NumPy (CPU) or CuPy (GPU). Typical uses include validating two stream growth rates, checking energy closure, fourier analysis and visualizing phase space evolution.

## How to Use
1. **Environment dependencies**
   - Required: `python >= 3.9`, `numpy`, `scipy`
   - GPU option: `cupy` (skip if running on CPU only)
   - Optional: `matplotlib`, `notebook` (for visualization in `diagnostics/Diag_visual.ipynb`)

2. **Install dependencies**
   ```bash
   pip install numpy scipy matplotlib notebook
   # For GPU acceleration, install a CuPy build matching your CUDA runtime, e.g.:
   pip install cupy-cuda12x
   ```

3. **Run the simulation**
   ```bash
   python main.py
   ```
   The run will clear/create the `output` directory and produce energy, spectrum, and phase-space files during the simulation.

4. **Select CPU or GPU backend**
   - Edit `source_code/backend.py` and set `FORCE_DEVICE`:
     - `"cpu"`: force NumPy execution.
     - `"gpu"`: force CuPy; raises an error if GPU is unavailable.
     - `"auto"`: prefer GPU and fall back to CPU (default).
   - Adjust `SEED` in the same file if you need deterministic runs.

5. **CPU/GPU debugging tips**
   - CPU: Set `FORCE_DEVICE = "cpu"` to simplify step-through debugging with any Python IDE.
   - GPU: Set `FORCE_DEVICE = "gpu"` and reduce `n_particles` or `n_steps` in `main.py` to shorten iteration cycles while profiling.

## Directory Structure
- `main.py`: Entry point that loads configuration, initializes the two-stream distribution, and runs the main loop.
- `source_code/`: Core numerical modules.
  - `backend.py`: Manages the NumPy/CuPy backend, random seeds, and helper utilities.
  - `dataclass.py`: Defines the `PICConfig` data class and computes normalized parameters.
  - `grid1D.py`: Declares the 1D grid and grid spacing.
  - `particles.py`: Container for particle properties and distributions.
  - `initial.py`: Helpers for initializing particle distributions.
  - `fields.py`: Holds potential, electric field, and related field quantities.
  - `deposit_charge.py`: Performs CIC charge deposition onto the grid.
  - `solve_poisson_1d.py`: Poisson solver with periodic boundary conditions.
  - `gather_field.py`: Interpolates grid fields to particle positions.
  - `push_particle.py`: Implements the (electrostatic) Boris pusher.
  - `diagnostics.py`: Computes and saves energy, spectra, and phase-space diagnostics.
  - `PIC1D3V_ES.py`: Main electrostatic PIC class that wires components, initializes state, advances in time, and triggers diagnostics.
  - `__init__.py`: Package initializer.
- `diagnostics/`: Analysis and visualization: including kinetic and electrostatic energy, linear growth rate calculation, fourier analysis, numerical solved dispersion relation, phase space evolution, velocity distribution evolution, position distribution evolution, the evolution of the electrostatic potential energy for a electron.
  - `Diag_visual.ipynb`: Example notebook for plotting diagnostic outputs.
  - `config_paths.py`, `diag_utils.py`: Path configuration plus plotting/processing utilities.
  - `gamma_vs_vth.png`: Sample growth-rate figure.
- `TEST_algotithm/`: Basic validation scripts and examples.
  - `test_gather_field.py`, `test_push.py`, `test_solve_poisson.py`: Simple checks for key modules.
  - `test_depoit_charge`: Example data for charge-deposition results.
  - `boris_uniformE_validation.png`: Validation figure for the Boris pusher.
- `__pycache__/`: Python bytecode cache generated at runtime.
- `README.md`: Project overview and usage (this file).
- `output/` (generated during runs): Stores energy, spectrum, and phase-space outputs.
