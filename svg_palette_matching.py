# svg_palette_matching.py
"""
(全新重构) 稳健的色板匹配模块，采用“三阶段锦标赛式”选择算法，确保选择的准确性、多样性和艺术效果。
"""
import numpy as np
from typing import List, Tuple, Dict
from collections import defaultdict
from config import CONFIG
import cv2

def rgb_to_lab(color_rgb: Tuple[int, int, int]) -> np.ndarray:
    """将RGB颜色安全地转换为LAB颜色空间"""
    if not isinstance(color_rgb, (list, tuple)) or len(color_rgb) != 3:
        return np.array([0, 0, 0], dtype=np.float64)
    rgb_uint8 = np.uint8([[list(color_rgb)]])
    return cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2LAB)[0][0].astype(float)

class SVGPaletteMatcher:
    def __init__(self):
        self.env_palette = CONFIG["env_palette"]
        self.skin_palettes = CONFIG["skin_palettes"]
        self.skin_palettes_lab = {name: [rgb_to_lab(c) for c in p] for name, p in self.skin_palettes.items()}
        self.env_palette_lab = [rgb_to_lab(c) for c in self.env_palette]

    def match_environment_colors(self, env_colors: List[Tuple[int, int, int]]) -> Dict[Tuple[int, int, int], Tuple[int, int, int]]:
        color_mapping = {}
        for color in env_colors:
            if not self.env_palette:
                color_mapping[color] = color
                continue
            color_lab = rgb_to_lab(color)
            distances = [cv2.norm(color_lab, p_lab, cv2.NORM_L2) for p_lab in self.env_palette_lab]
            color_mapping[color] = tuple(self.env_palette[np.argmin(distances)])
        return color_mapping

    def match_skin_colors(self,
                         original_skin_colors: List[Tuple[int, int, int]],
                         quantized_skin_colors: List[Tuple[int, int, int]]) -> Tuple[Dict[Tuple[int, int, int], Tuple[int, int, int]], str]:
        if not original_skin_colors or not self.skin_palettes:
            return {c: c for c in quantized_skin_colors}, "No Palettes"

        best_palette_name = self._select_best_skin_palette_robust(original_skin_colors)

        best_palette_lab = self.skin_palettes_lab.get(best_palette_name, [])
        best_palette_rgb = self.skin_palettes.get(best_palette_name, [])

        if not best_palette_lab:
            return {c: c for c in quantized_skin_colors}, best_palette_name

        color_mapping = {}
        quantized_lab = [rgb_to_lab(c) for c in quantized_skin_colors]
        for i, color in enumerate(quantized_skin_colors):
            distances = [cv2.norm(quantized_lab[i], p_lab, cv2.NORM_L2) for p_lab in best_palette_lab]
            color_mapping[color] = tuple(best_palette_rgb[np.argmin(distances)])

        return color_mapping, best_palette_name

    def _get_lab_stats(self, lab_colors: List[np.ndarray]) -> Dict:
        """计算LAB颜色列表的统计数据"""
        if not lab_colors:
            return {'mean_l': 50, 'mean_a': 0, 'mean_b': 0, 'std_l': 0, 'std_a': 0, 'std_b': 0}
        
        l, a, b = zip(*lab_colors)
        return {
            'mean_l': np.mean(l), 'mean_a': np.mean(a), 'mean_b': np.mean(b),
            'std_l': np.std(l), 'std_a': np.std(a), 'std_b': np.std(b)
        }

    def _select_best_skin_palette_robust(self, original_skin_colors: List[Tuple[int, int, int]]) -> str:
        """
        【全新三阶段锦标赛算法】
        1. 海选淘汰：剔除明显不符的色板。
        2. 分组对抗：通过模拟匹配计算保真度。
        3. 决赛裁定：处理平局，选择最优方案。
        """
        if not original_skin_colors: return next(iter(self.skin_palettes), "SKIN-A")

        input_lab = [rgb_to_lab(c) for c in original_skin_colors]
        input_stats = self._get_lab_stats(input_lab)

        # --- 第一阶段：海选淘汰 ---
        candidate_palettes = []
        for name, palette_lab in self.skin_palettes_lab.items():
            if not palette_lab: continue
            
            palette_stats = self._get_lab_stats(palette_lab)

            # 淘汰标准1: 亮度范围差异过大
            if abs(input_stats['mean_l'] - palette_stats['mean_l']) > 35:
                continue
            # 淘汰标准2: 色温方向相反 (b*通道 > 0 偏黄/暖, < 0 偏蓝/冷)
            if np.sign(input_stats['mean_b']) != np.sign(palette_stats['mean_b']) and abs(input_stats['mean_b']) > 5:
                continue
            # 淘汰标准3: 动态范围(亮度标准差)严重不匹配
            if input_stats['std_l'] > 10 and palette_stats['std_l'] < input_stats['std_l'] * 0.4:
                continue

            candidate_palettes.append(name)
        
        # 如果所有色板都被淘汰，则放宽标准重新海选
        if not candidate_palettes:
            candidate_palettes = list(self.skin_palettes.keys())

        # --- 第二阶段：分组对抗，计算保真度 ---
        fidelity_scores = {}
        for name in candidate_palettes:
            palette_lab = self.skin_palettes_lab[name]
            total_error = 0
            for color_lab in input_lab:
                distances = [cv2.norm(color_lab, p_lab, cv2.NORM_L2) for p_lab in palette_lab]
                total_error += min(distances)
            # 保真度 = 1 / (1 + 平均误差)，误差越小，保真度越高
            fidelity_scores[name] = 1.0 / (1.0 + total_error / len(input_lab))

        # --- 第三阶段：决赛裁定 ---
        sorted_scores = sorted(fidelity_scores.items(), key=lambda x: x[1], reverse=True)
        if not sorted_scores:
            return next(iter(self.skin_palettes), "SKIN-A")

        best_palette_name = sorted_scores[0][0]
        
        # 如果最高分和次高分非常接近，则选择细节更丰富的那个
        if len(sorted_scores) > 1 and (sorted_scores[0][1] - sorted_scores[1][1]) < 0.02: # 保真度差异小于2%
            top1_name, top2_name = sorted_scores[0][0], sorted_scores[1][0]
            top1_stats = self._get_lab_stats(self.skin_palettes_lab[top1_name])
            top2_stats = self._get_lab_stats(self.skin_palettes_lab[top2_name])
            # 选择色彩更丰富(a*和b*标准差之和更大)的那个
            if (top2_stats['std_a'] + top2_stats['std_b']) > (top1_stats['std_a'] + top1_stats['std_b']):
                best_palette_name = top2_name
        
        return best_palette_name

    def match_svg_colors(self,
                        quantized_mapping: Dict[int, Tuple[int, int, int]],
                        skin_elements: List[Tuple[int, Tuple[int, int, int]]],
                        env_elements: List[Tuple[int, Tuple[int, int, int]]]) -> Dict[int, Tuple[int, int, int]]:
        final_mapping = quantized_mapping.copy()

        original_skin_colors = list(set(color for _, color in skin_elements))
        quantized_indices = set(quantized_mapping.keys())
        quantized_skin_colors = list(set(quantized_mapping[idx] for idx, _ in skin_elements if idx in quantized_indices))
        quantized_env_colors = list(set(quantized_mapping[idx] for idx, _ in env_elements if idx in quantized_indices))

        skin_map, palette_name = self.match_skin_colors(original_skin_colors, quantized_skin_colors)
        print(f"选择的皮肤色板: {palette_name} (基于三阶段锦标赛算法)")
        env_map = self.match_environment_colors(quantized_env_colors)

        for idx, old_quantized_color in quantized_mapping.items():
            if old_quantized_color in skin_map:
                final_mapping[idx] = skin_map[old_quantized_color]
            elif old_quantized_color in env_map:
                final_mapping[idx] = env_map[old_quantized_color]

        return final_mapping