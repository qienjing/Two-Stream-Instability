# diagnostics/config_paths.py
"""
全局路径配置文件
所有诊断脚本通过 import 这个文件获得路径定义
"""
import os

# --- 自动定位项目根目录 ---
THIS_DIR = os.path.dirname(os.path.abspath(__file__))

# 假设结构为：
# project_root/
# ├── output/
# └── diagnostics/
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))

# --- 数据输出路径 ---
OUT = os.path.join(PROJECT_ROOT, "output")
if not os.path.isdir(OUT):
    raise FileNotFoundError(f"❌ 未找到 output 文件夹: {OUT}")

# --- 图像/分析结果保存路径 ---
FIGS = os.path.join(OUT, "figs")
os.makedirs(FIGS, exist_ok=True)

print(f"✅ [config_paths] 输出目录: {OUT}")
print(f"✅ [config_paths] 图像目录: {FIGS}")
