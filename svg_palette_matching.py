# svg_palette_matching.py
"""
增强的色板匹配模块，用于SVG直接颜色匹配
采纳了基于平均亮度和严格顺序映射的优化算法
"""
import numpy as np
from typing import List, Tuple, Dict, Optional
from config import CONFIG

def brightness(color: Tuple[int, int, int]) -> float:
    """计算颜色亮度"""
    # 确保颜色值是数值类型
    r, g, b = color
    return 0.299 * float(r) + 0.587 * float(g) + 0.114 * float(b)

def euclidean_distance(c1: Tuple[int, int, int], c2: Tuple[int, int, int]) -> float:
    """计算两个颜色之间的欧氏距离"""
    return np.linalg.norm(np.array(c1, dtype=np.float32) - np.array(c2, dtype=np.float32))

class SVGPaletteMatcher:
    """SVG色板匹配器"""
    
    def __init__(self):
        self.env_palette = CONFIG["env_palette"]
        self.skin_palettes = CONFIG["skin_palettes"]
    
    def match_environment_colors(self, 
                                env_colors: List[Tuple[int, int, int]]) -> Dict[Tuple[int, int, int], Tuple[int, int, int]]:
        """环境区域颜色匹配 - 简单最近邻"""
        color_mapping = {}
        for color in env_colors:
            if color not in color_mapping:
                min_dist = float('inf')
                best_match = self.env_palette[0] if self.env_palette else color
                for palette_color in self.env_palette:
                    palette_color_tuple = tuple(int(c) for c in palette_color)
                    dist = euclidean_distance(color, palette_color_tuple)
                    if dist < min_dist:
                        min_dist = dist
                        best_match = palette_color_tuple
                color_mapping[color] = best_match
        return color_mapping
    
    def match_skin_colors(self, 
                         quantized_colors: List[Tuple[int, int, int]],
                         original_skin_colors: List[Tuple[int, int, int]]) -> Tuple[Dict[Tuple[int, int, int], Tuple[int, int, int]], str]:
        """
        皮肤区域颜色匹配 - 采用基于平均亮度和严格顺序的先进算法
        """
        if not quantized_colors:
            return {}, ""
        
        # 1. 选择最佳皮肤色板
        best_palette_name, best_palette = self._select_best_skin_palette(original_skin_colors)
        
        if not best_palette:
            return {c: c for c in quantized_colors}, ""

        # 2. 排序
        # 对量化后的颜色按亮度排序
        sorted_quantized_colors = sorted(list(set(quantized_colors)), key=brightness)
        # 对选定的色板颜色按亮度排序
        sorted_palette = sorted(best_palette, key=lambda c: brightness(tuple(c)))
        
        # 3. 创建映射
        color_mapping = self._create_sequential_mapping(sorted_quantized_colors, sorted_palette)
        
        return color_mapping, best_palette_name

    def _select_best_skin_palette(self, 
                                 original_colors: List[Tuple[int, int, int]]) -> Tuple[str, List[Tuple[int,int,int]]]:
        """根据原始皮肤颜色的平均亮度选择最匹配的色板"""
        if not original_colors:
            # 如果没有原始颜色，则退回使用第一个色板作为默认
            first_palette_name = next(iter(self.skin_palettes))
            return first_palette_name, self.skin_palettes.get(first_palette_name, [])

        # 计算原始皮肤颜色的平均亮度
        avg_brightness = np.mean([brightness(c) for c in original_colors])
        
        best_palette_name = ""
        best_palette = []
        min_diff = float('inf')
        
        for name, palette in self.skin_palettes.items():
            if not palette:
                continue
            palette_brightness = np.mean([brightness(tuple(c)) for c in palette])
            diff = abs(palette_brightness - avg_brightness)
            if diff < min_diff:
                min_diff = diff
                best_palette_name = name
                best_palette = palette
        
        return best_palette_name, best_palette

    def _create_sequential_mapping(self, 
                                  sorted_colors: List[Tuple[int, int, int]], 
                                  sorted_palette: List) -> Dict[Tuple[int, int, int], Tuple[int, int, int]]:
        """创建保持顺序的颜色映射"""
        n_colors = len(sorted_colors)
        n_palette = len(sorted_palette)
        mapping = {}
        
        if n_colors == 0 or n_palette == 0:
            return mapping
            
        # 将聚类中心映射到色板
        for i, color in enumerate(sorted_colors):
            # 计算在色板中的理想位置比例
            ratio = i / (n_colors - 1) if n_colors > 1 else 0
            # 找到色板中对应的索引
            palette_idx = int(round(ratio * (n_palette - 1)))
            palette_color = tuple(int(c) for c in sorted_palette[palette_idx])
            mapping[color] = palette_color
            
        return mapping

    def match_svg_colors(self,
                        skin_color_groups: List[Tuple[List[int], Tuple[int, int, int]]],
                        env_color_groups: List[Tuple[List[int], Tuple[int, int, int]]],
                        original_skin_elements: List[Tuple[int, Tuple[int, int, int]]]) -> Dict[int, Tuple[int, int, int]]:
        """主匹配函数：将SVG元素颜色映射到色板"""
        element_color_mapping = {}
        
        # 1. 处理皮肤区域
        if skin_color_groups:
            quantized_skin_colors = [color for _, color in skin_color_groups]
            original_skin_colors = [color for _, color in original_skin_elements]
            
            # 调用新的皮肤匹配算法
            skin_map, palette_name = self.match_skin_colors(quantized_skin_colors, original_skin_colors)
            print(f"选择的皮肤色板: {palette_name} (基于原始颜色亮度)")
            
            for indices, color in skin_color_groups:
                new_color = skin_map.get(color, color)
                for idx in indices:
                    element_color_mapping[idx] = new_color
        
        # 2. 处理环境区域
        if env_color_groups:
            env_colors = [color for _, color in env_color_groups]
            env_map = self.match_environment_colors(env_colors)
            
            for indices, color in env_color_groups:
                new_color = env_map.get(color, color)
                for idx in indices:
                    element_color_mapping[idx] = new_color
        
        return element_color_mapping