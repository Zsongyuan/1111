# svg_palette_matching.py
"""
增强的色板匹配模块，用于SVG直接颜色匹配
"""
import numpy as np
from typing import List, Tuple, Dict, Optional
from config import CONFIG

def brightness(color: Tuple[int, int, int]) -> float:
    """计算颜色亮度"""
    return 0.299 * color[0] + 0.587 * color[1] + 0.114 * color[2]

def euclidean_distance(c1: Tuple[int, int, int], c2: Tuple[int, int, int]) -> float:
    """计算两个颜色之间的欧氏距离"""
    return np.sqrt(sum((a - b) ** 2 for a, b in zip(c1, c2)))

class SVGPaletteMatcher:
    """SVG色板匹配器"""
    
    def __init__(self):
        self.env_palette = CONFIG["env_palette"]
        self.skin_palettes = CONFIG["skin_palettes"]
    
    def match_environment_colors(self, 
                                env_colors: List[Tuple[int, int, int]]) -> Dict[Tuple[int, int, int], Tuple[int, int, int]]:
        """
        环境区域颜色匹配 - 简单最近邻
        
        参数:
            env_colors: 环境区域的颜色列表
            
        返回:
            原色到新色的映射
        """
        color_mapping = {}
        
        for color in env_colors:
            if color not in color_mapping:
                # 找到最近的环境色板颜色
                min_dist = float('inf')
                best_match = self.env_palette[0] if self.env_palette else color
                
                for palette_color in self.env_palette:
                    # 确保色板颜色是整数元组
                    if isinstance(palette_color, list):
                        palette_color = tuple(int(c) for c in palette_color)
                    dist = euclidean_distance(color, palette_color)
                    if dist < min_dist:
                        min_dist = dist
                        best_match = palette_color
                
                color_mapping[color] = best_match
        
        return color_mapping
    
    def match_skin_colors(self, 
                         skin_colors: List[Tuple[int, int, int]],
                         original_skin_colors: Optional[List[Tuple[int, int, int]]] = None) -> Tuple[Dict[Tuple[int, int, int], Tuple[int, int, int]], str]:
        """
        皮肤区域颜色匹配 - 保持顺序约束
        
        参数:
            skin_colors: 量化后的皮肤颜色列表
            original_skin_colors: 原始皮肤颜色（用于选择最佳色板）
            
        返回:
            (颜色映射, 选择的色板名称)
        """
        if not skin_colors:
            return {}, ""
        
        # 去重
        unique_colors = list(set(skin_colors))
        
        # 选择最佳皮肤色板
        best_palette_name, best_palette = self._select_best_skin_palette(
            unique_colors, original_skin_colors
        )
        
        if not best_palette:
            return {c: c for c in unique_colors}, ""
        
        # 按亮度排序
        sorted_colors = sorted(unique_colors, key=brightness)
        sorted_palette = sorted(best_palette, key=lambda c: brightness(tuple(c)))
        
        # 创建映射
        color_mapping = self._create_sequential_mapping(sorted_colors, sorted_palette)
        
        return color_mapping, best_palette_name
    
    def _select_best_skin_palette(self, 
                                 quantized_colors: List[Tuple[int, int, int]],
                                 original_colors: Optional[List[Tuple[int, int, int]]]) -> Tuple[str, List]:
        """选择最匹配的皮肤色板"""
        # 计算参考亮度
        if original_colors:
            avg_brightness = np.mean([brightness(c) for c in original_colors])
        else:
            avg_brightness = np.mean([brightness(c) for c in quantized_colors])
        
        best_palette_name = ""
        best_palette = []
        best_diff = float('inf')
        
        for name, palette in self.skin_palettes.items():
            if not palette:
                continue
            
            # 计算色板平均亮度
            palette_brightness = np.mean([brightness(tuple(c)) for c in palette])
            diff = abs(palette_brightness - avg_brightness)
            
            if diff < best_diff:
                best_diff = diff
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
        
        if n_colors <= n_palette:
            # 颜色数少于或等于色板数，直接映射
            for i, color in enumerate(sorted_colors):
                # 计算在色板中的位置
                palette_idx = int(i * n_palette / n_colors)
                # 确保色板颜色是整数元组
                palette_color = sorted_palette[palette_idx]
                if isinstance(palette_color, list):
                    palette_color = tuple(int(c) for c in palette_color)
                mapping[color] = palette_color
        else:
            # 颜色数多于色板数，分段映射
            colors_per_palette = n_colors / n_palette
            
            for i, color in enumerate(sorted_colors):
                palette_idx = min(int(i / colors_per_palette), n_palette - 1)
                # 确保色板颜色是整数元组
                palette_color = sorted_palette[palette_idx]
                if isinstance(palette_color, list):
                    palette_color = tuple(int(c) for c in palette_color)
                mapping[color] = palette_color
        
        return mapping
    
    def match_svg_colors(self,
                        skin_color_groups: List[Tuple[List[int], Tuple[int, int, int]]],
                        env_color_groups: List[Tuple[List[int], Tuple[int, int, int]]],
                        original_skin_elements: List[Tuple[int, Tuple[int, int, int]]]) -> Dict[int, Tuple[int, int, int]]:
        """
        匹配SVG元素颜色到色板
        
        参数:
            skin_color_groups: [(元素索引列表, 聚类颜色), ...]
            env_color_groups: [(元素索引列表, 聚类颜色), ...]
            original_skin_elements: 原始皮肤元素列表（用于色板选择）
            
        返回:
            元素索引到最终颜色的映射
        """
        element_color_mapping = {}
        
        # 处理皮肤区域
        if skin_color_groups:
            skin_colors = [color for _, color in skin_color_groups]
            original_colors = [color for _, color in original_skin_elements] if original_skin_elements else None
            
            color_mapping, palette_name = self.match_skin_colors(skin_colors, original_colors)
            print(f"选择的皮肤色板: {palette_name}")
            
            # 应用映射到元素
            for indices, color in skin_color_groups:
                new_color = color_mapping.get(color, color)
                for idx in indices:
                    element_color_mapping[idx] = new_color
        
        # 处理环境区域
        if env_color_groups:
            env_colors = [color for _, color in env_color_groups]
            color_mapping = self.match_environment_colors(env_colors)
            
            # 应用映射到元素
            for indices, color in env_color_groups:
                new_color = color_mapping.get(color, color)
                for idx in indices:
                    element_color_mapping[idx] = new_color
        
        return element_color_mapping