# diagnostics/config_paths.py
"""
Global path configuration file.
All diagnostic scripts import this module to get path definitions.
"""
import os

# --- Auto-locate project root ---
THIS_DIR = os.path.dirname(os.path.abspath(__file__))

PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))

# --- Data output path ---
OUT = os.path.join(PROJECT_ROOT, "output")
if not os.path.isdir(OUT):
    raise FileNotFoundError(f"❌ Output folder not found: {OUT}")

# --- Figure/analysis output path ---
FIGS = os.path.join(OUT, "figs")
os.makedirs(FIGS, exist_ok=True)

print(f"✅ [config_paths] Output directory: {OUT}")
print(f"✅ [config_paths] Figures directory: {FIGS}")
