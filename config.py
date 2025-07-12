# config.py

import pandas as pd
import os
import glob

# 定义输入和输出目录（请确保这两个文件夹存在或在运行时创建）
INPUT_DIR = os.path.join(".", "input")
OUTPUT_DIR = os.path.join(".", "output4")

# 定义色板文件的路径
ENV_PALETTE_PATH = os.path.join(".", "palette", "envPalettes", "ENV.xls")
SKIN_PALETTE_DIR = os.path.join(".", "palette", "skinPalettes")

def load_environment_palette():
    """
    读取环境色板文件，返回色板列表，每个元素为 [R, G, B]。
    假设文件中包含 "Year", "R", "G", "B" 列，并且已按颜色从浅到深排序。
    对于存在 NaN 的行直接剔除。
    """
    if not os.path.exists(ENV_PALETTE_PATH):
        raise FileNotFoundError(f"环境色板文件未找到: {ENV_PALETTE_PATH}")
    df = pd.read_excel(ENV_PALETTE_PATH)
    df = df.dropna(subset=['R', 'G', 'B'])
    palette = df[['R', 'G', 'B']].values.tolist()
    return palette

def load_skin_palettes():
    """
    读取皮肤色板文件，返回一个字典，
    键为文件名（不含扩展名，如 "SKIN-A"），值为色板列表，每个元素为 [R, G, B]。
    对于存在 NaN 的行直接剔除。
    """
    skin_palette_files = glob.glob(os.path.join(SKIN_PALETTE_DIR, "SKIN-*.xlsx"))
    skin_palettes = {}
    for file in skin_palette_files:
        key = os.path.splitext(os.path.basename(file))[0]  # 例如 "SKIN-A"
        df = pd.read_excel(file)
        df = df.dropna(subset=['R', 'G', 'B'])
        palette = df[['R', 'G', 'B']].values.tolist()
        skin_palettes[key] = palette
    return skin_palettes

# 全局配置字典，可供其它模块直接调用
CONFIG = {
    "INPUT_DIR": INPUT_DIR,
    "OUTPUT_DIR": OUTPUT_DIR,
    "env_palette": load_environment_palette(),
    "skin_palettes": load_skin_palettes(),
    # 其他全局参数（如API密钥等）可在此扩展
}

if __name__ == '__main__':
    print("输入目录:", CONFIG["INPUT_DIR"])
    print("输出目录:", CONFIG["OUTPUT_DIR"])
    print("环境色板:")
    print(CONFIG["env_palette"])
    print("皮肤色板:")
    for key, palette in CONFIG["skin_palettes"].items():
        print(f"{key}: {palette}")
