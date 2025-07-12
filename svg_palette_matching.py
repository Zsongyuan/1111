# svg_palette_matching.py
"""
增强的色板匹配模块，采用基于色彩直方图的先进算法，确保色板选择的准确性。
"""
import numpy as np
from typing import List, Tuple, Dict
from config import CONFIG
import cv2

def rgb_to_lab(color_rgb: Tuple[int, int, int]) -> np.ndarray:
    rgb_uint8 = np.uint8([[list(color_rgb)]])
    return cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2LAB)[0][0].astype(float)

def delta_e_2000(lab1: np.ndarray, lab2: np.ndarray) -> float:
    return cv2.norm(lab1, lab2, cv2.NORM_L2)

# 【核心新增】
def get_lab_color_histogram(lab_colors: List[np.ndarray], bins=8) -> np.ndarray:
    """计算LAB颜色列表的3D直方图"""
    if not lab_colors:
        return np.zeros((bins, bins, bins))
    
    # LAB的范围: L(0-100), a(-128-127), b(-128-127)
    # 我们将范围归一化以便创建直方图
    l_chan = [c[0] for c in lab_colors]
    a_chan = [c[1] for c in lab_colors]
    b_chan = [c[2] for c in lab_colors]
    
    hist, _ = np.histogramdd(
        (l_chan, a_chan, b_chan),
        bins=(bins, bins, bins),
        range=((0, 100), (-128, 127), (-128, 127)),
        density=True # 使用密度以忽略颜色数量的影响
    )
    return hist

class SVGPaletteMatcher:
    def __init__(self):
        self.env_palette = CONFIG["env_palette"]
        self.skin_palettes = CONFIG["skin_palettes"]
        self.skin_palettes_lab = {name: [rgb_to_lab(c) for c in p] for name, p in self.skin_palettes.items()}
        self.env_palette_lab = [rgb_to_lab(c) for c in self.env_palette]
        # 【新增】预计算色板的直方图
        self.skin_histograms = {name: get_lab_color_histogram(p_lab) for name, p_lab in self.skin_palettes_lab.items()}

    def match_environment_colors(self, env_colors: List[Tuple[int, int, int]]) -> Dict[Tuple[int, int, int], Tuple[int, int, int]]:
        color_mapping = {}
        for color in env_colors:
            if not self.env_palette: color_mapping[color] = color; continue
            color_lab = rgb_to_lab(color)
            distances = [delta_e_2000(color_lab, p_lab) for p_lab in self.env_palette_lab]
            color_mapping[color] = tuple(self.env_palette[np.argmin(distances)])
        return color_mapping

    def match_skin_colors(self, 
                         quantized_colors: List[Tuple[int, int, int]]) -> Tuple[Dict[Tuple[int, int, int], Tuple[int, int, int]], str]:
        if not quantized_colors or not self.skin_palettes:
            return {c: c for c in quantized_colors}, "No Palettes"
        
        # 1. 【核心重构】使用直方图选择最佳色板
        quantized_lab = [rgb_to_lab(c) for c in quantized_colors]
        best_palette_name = self._select_best_skin_palette_by_hist(quantized_lab)
        
        best_palette_lab = self.skin_palettes_lab.get(best_palette_name, [])
        best_palette_rgb = self.skin_palettes.get(best_palette_name, [])
        
        if not best_palette_lab: return {c:c for c in quantized_colors}, best_palette_name

        # 2. 精确映射：对每个颜色找到最佳匹配
        color_mapping = {}
        for i, color in enumerate(quantized_colors):
            distances = [delta_e_2000(quantized_lab[i], p_lab) for p_lab in best_palette_lab]
            color_mapping[color] = tuple(best_palette_rgb[np.argmin(distances)])
            
        return color_mapping, best_palette_name

    # 【核心重构】
    def _select_best_skin_palette_by_hist(self, quantized_lab_colors: List[np.ndarray]) -> str:
        """通过比较色彩直方图的相似度来选择最佳色板"""
        if not quantized_lab_colors: return next(iter(self.skin_palettes), "")

        # 1. 计算输入颜色的直方图
        input_hist = get_lab_color_histogram(quantized_lab_colors)

        best_palette_name = ""
        best_score = -1

        # 2. 将输入直方图与每个备选色板的预计算直方图进行比较
        for name, palette_hist in self.skin_histograms.items():
            # 使用“相关性”作为比较方法，得分越高越相似
            score = cv2.compareHist(input_hist.astype(np.float32), palette_hist.astype(np.float32), cv2.HISTCMP_CORREL)
            if score > best_score:
                best_score = score
                best_palette_name = name
        
        return best_palette_name

    def match_svg_colors(self,
                        quantized_mapping: Dict[int, Tuple[int, int, int]],
                        skin_elements: List[Tuple[int, Tuple[int, int, int]]],
                        env_elements: List[Tuple[int, Tuple[int, int, int]]]) -> Dict[int, Tuple[int, int, int]]:
        final_mapping = quantized_mapping.copy()
        
        quantized_indices = set(quantized_mapping.keys())
        skin_elements_in_map = [(idx, color) for idx, color in skin_elements if idx in quantized_indices]
        env_elements_in_map = [(idx, color) for idx, color in env_elements if idx in quantized_indices]

        # 提取需要匹配的独特颜色
        skin_colors_to_match = list(set(c for _, c in skin_elements_in_map))
        env_colors_to_match = list(set(c for _, c in env_elements_in_map))

        # 执行匹配
        skin_map, palette_name = self.match_skin_colors(skin_colors_to_match)
        print(f"选择的皮肤色板: {palette_name} (基于色彩直方图)")
        env_map = self.match_environment_colors(env_colors_to_match)

        # 应用匹配结果
        for idx, old_color in quantized_mapping.items():
            if old_color in skin_map:
                final_mapping[idx] = skin_map[old_color]
            elif old_color in env_map:
                final_mapping[idx] = env_map[old_color]

        return final_mapping