# svg_palette_matching.py
"""
稳健版色板匹配模块：基于实际颜色匹配质量选择色板
"""
import numpy as np
from typing import List, Tuple, Dict
from config import CONFIG
import cv2

def rgb_to_lab(color_rgb: Tuple[int, int, int]) -> np.ndarray:
    rgb_uint8 = np.uint8([[list(color_rgb)]])
    return cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2LAB)[0][0].astype(float)

def delta_e_2000(lab1: np.ndarray, lab2: np.ndarray) -> float:
    """计算LAB色彩空间中的感知距离"""
    return np.sqrt(np.sum((lab1 - lab2) ** 2))

def calculate_color_temperature(colors: List[Tuple[int, int, int]]) -> float:
    """计算颜色的平均色温倾向"""
    if not colors:
        return 0.0
    
    temp_scores = []
    for r, g, b in colors:
        # 简化的色温计算：蓝色分量越高色温越高（冷色），红色分量越高色温越低（暖色）
        temp_score = (b - r) / 255.0  # 范围 -1 到 1
        temp_scores.append(temp_score)
    
    return np.mean(temp_scores)

class RobustSVGPaletteMatcher:
    """基于实际匹配质量的稳健色板选择器"""
    
    def __init__(self):
        self.env_palette = CONFIG["env_palette"]
        self.skin_palettes = CONFIG["skin_palettes"]
        self.skin_palettes_lab = {name: [rgb_to_lab(c) for c in p] for name, p in self.skin_palettes.items()}
        self.env_palette_lab = [rgb_to_lab(c) for c in self.env_palette]
        
        print(f"加载了 {len(self.skin_palettes)} 个皮肤色板")

    def match_environment_colors(self, env_colors: List[Tuple[int, int, int]]) -> Dict[Tuple[int, int, int], Tuple[int, int, int]]:
        """匹配环境颜色到环境色板"""
        color_mapping = {}
        unique_env_colors = list(set(env_colors))
        
        for color in unique_env_colors:
            if not self.env_palette: 
                color_mapping[color] = color
                continue
                
            color_lab = rgb_to_lab(color)
            distances = [delta_e_2000(color_lab, p_lab) for p_lab in self.env_palette_lab]
            matched_color = self.env_palette[np.argmin(distances)]
            color_mapping[color] = tuple(int(c) for c in matched_color)
        
        # 为所有原始颜色建立映射
        final_mapping = {}
        for color in env_colors:
            final_mapping[color] = color_mapping[color]
            
        return final_mapping

    def match_skin_colors(self, 
                         quantized_colors: List[Tuple[int, int, int]],
                         original_skin_colors: List[Tuple[int, int, int]] = None) -> Tuple[Dict[Tuple[int, int, int], Tuple[int, int, int]], str]:
        """基于匹配质量的皮肤颜色匹配"""
        if not quantized_colors or not self.skin_palettes:
            return {c: c for c in quantized_colors}, "No Palettes"
        
        # 使用原始皮肤颜色来选择色板（如果提供）
        colors_for_selection = original_skin_colors if original_skin_colors else quantized_colors
        best_palette_name = self._select_best_skin_palette_robust(colors_for_selection)
        
        print(f"色板选择结果: {best_palette_name}")
        
        best_palette_lab = self.skin_palettes_lab.get(best_palette_name, [])
        best_palette_rgb = self.skin_palettes.get(best_palette_name, [])
        
        if not best_palette_lab: 
            return {c: c for c in quantized_colors}, best_palette_name

        # 精确的一对一映射
        color_mapping = {}
        unique_quantized = list(set(quantized_colors))
        
        for color in unique_quantized:
            color_lab = rgb_to_lab(color)
            distances = [delta_e_2000(color_lab, p_lab) for p_lab in best_palette_lab]
            best_match_idx = np.argmin(distances)
            matched_color = best_palette_rgb[best_match_idx]
            color_mapping[color] = tuple(int(c) for c in matched_color)
        
        # 为所有原始颜色建立映射
        final_mapping = {}
        for color in quantized_colors:
            final_mapping[color] = color_mapping[color]
            
        return final_mapping, best_palette_name

    def _select_best_skin_palette_robust(self, input_colors: List[Tuple[int, int, int]]) -> str:
        """稳健的皮肤色板选择算法 - 基于实际匹配质量"""
        if not input_colors:
            # 智能回退：优先选择中等肤色色板
            fallback_order = ["SKIN-H", "SKIN-I", "SKIN-J", "SKIN-K", "SKIN-G", "SKIN-L"]
            for palette in fallback_order:
                if palette in self.skin_palettes:
                    print(f"使用回退色板: {palette}")
                    return palette
            return next(iter(self.skin_palettes), "")

        print(f"\n开始色板选择，输入颜色数: {len(input_colors)}")
        
        # 转换输入颜色到LAB空间
        input_lab_colors = [rgb_to_lab(c) for c in input_colors]
        
        # 方法1: 基于最佳匹配质量的评分
        method1_scores = self._evaluate_by_matching_quality(input_lab_colors)
        
        # 方法2: 基于色彩覆盖度的评分
        method2_scores = self._evaluate_by_coverage_quality(input_colors)
        
        # 方法3: 基于色温匹配的评分
        method3_scores = self._evaluate_by_color_temperature(input_colors)
        
        # 综合评分（可调整权重）
        final_scores = {}
        weight1, weight2, weight3 = 0.5, 0.3, 0.2  # 最佳匹配质量权重最高
        
        all_palettes = set(method1_scores.keys()) | set(method2_scores.keys()) | set(method3_scores.keys())
        
        print(f"\n各方法评分结果:")
        for palette in all_palettes:
            score1 = method1_scores.get(palette, 0)
            score2 = method2_scores.get(palette, 0)
            score3 = method3_scores.get(palette, 0)
            
            final_score = score1 * weight1 + score2 * weight2 + score3 * weight3
            final_scores[palette] = final_score
            
            print(f"  {palette}: 匹配={score1:.2f}, 覆盖={score2:.2f}, 色温={score3:.2f} -> 综合={final_score:.2f}")
        
        # 选择得分最高的色板
        best_palette = max(final_scores.items(), key=lambda x: x[1])
        print(f"\n最佳选择: {best_palette[0]} (得分: {best_palette[1]:.2f})")
        
        return best_palette[0]

    def _evaluate_by_matching_quality(self, input_lab_colors: List[np.ndarray]) -> Dict[str, float]:
        """方法1: 基于最佳匹配质量评分"""
        scores = {}
        
        for palette_name, palette_lab_colors in self.skin_palettes_lab.items():
            if not palette_lab_colors:
                scores[palette_name] = 0.0
                continue
            
            total_distance = 0.0
            for input_lab in input_lab_colors:
                # 找到该输入颜色在当前色板中的最佳匹配
                distances = [delta_e_2000(input_lab, p_lab) for p_lab in palette_lab_colors]
                min_distance = min(distances)
                total_distance += min_distance
            
            # 平均距离越小，评分越高（转换为0-1范围）
            avg_distance = total_distance / len(input_lab_colors)
            # 使用指数衰减函数，距离越小得分越高
            score = np.exp(-avg_distance / 50.0)  # 50是调节参数
            scores[palette_name] = score
        
        return scores

    def _evaluate_by_coverage_quality(self, input_colors: List[Tuple[int, int, int]]) -> Dict[str, float]:
        """方法2: 基于色彩覆盖度评分"""
        scores = {}
        
        for palette_name, palette_colors in self.skin_palettes.items():
            if not palette_colors:
                scores[palette_name] = 0.0
                continue
            
            # 计算输入颜色的色彩范围
            input_lab = [rgb_to_lab(c) for c in input_colors]
            palette_lab = [rgb_to_lab(c) for c in palette_colors]
            
            # 计算色板是否能良好覆盖输入颜色的范围
            input_l_range = [min(lab[0] for lab in input_lab), max(lab[0] for lab in input_lab)]
            input_a_range = [min(lab[1] for lab in input_lab), max(lab[1] for lab in input_lab)]
            input_b_range = [min(lab[2] for lab in input_lab), max(lab[2] for lab in input_lab)]
            
            palette_l_range = [min(lab[0] for lab in palette_lab), max(lab[0] for lab in palette_lab)]
            palette_a_range = [min(lab[1] for lab in palette_lab), max(lab[1] for lab in palette_lab)]
            palette_b_range = [min(lab[2] for lab in palette_lab), max(lab[2] for lab in palette_lab)]
            
            # 计算覆盖度
            l_coverage = self._calculate_range_overlap(input_l_range, palette_l_range)
            a_coverage = self._calculate_range_overlap(input_a_range, palette_a_range)
            b_coverage = self._calculate_range_overlap(input_b_range, palette_b_range)
            
            # 综合覆盖度评分
            coverage_score = (l_coverage + a_coverage + b_coverage) / 3.0
            scores[palette_name] = coverage_score
        
        return scores

    def _evaluate_by_color_temperature(self, input_colors: List[Tuple[int, int, int]]) -> Dict[str, float]:
        """方法3: 基于色温匹配评分"""
        scores = {}
        
        input_temp = calculate_color_temperature(input_colors)
        
        for palette_name, palette_colors in self.skin_palettes.items():
            if not palette_colors:
                scores[palette_name] = 0.0
                continue
            
            palette_temp = calculate_color_temperature(palette_colors)
            temp_diff = abs(input_temp - palette_temp)
            
            # 色温差异越小，得分越高
            score = np.exp(-temp_diff * 2.0)  # 2.0是调节参数
            scores[palette_name] = score
        
        return scores

    def _calculate_range_overlap(self, range1: List[float], range2: List[float]) -> float:
        """计算两个范围的重叠度"""
        min1, max1 = range1
        min2, max2 = range2
        
        # 计算重叠区间
        overlap_min = max(min1, min2)
        overlap_max = min(max1, max2)
        
        if overlap_min >= overlap_max:
            return 0.0  # 无重叠
        
        overlap_length = overlap_max - overlap_min
        total_range = max(max1, max2) - min(min1, min2)
        
        if total_range == 0:
            return 1.0
        
        return overlap_length / total_range

    def match_svg_colors(self,
                        quantized_mapping: Dict[int, Tuple[int, int, int]],
                        skin_elements: List[Tuple[int, Tuple[int, int, int]]],
                        env_elements: List[Tuple[int, Tuple[int, int, int]]],
                        original_skin_colors: List[Tuple[int, int, int]] = None,
                        target_k: int = None) -> Dict[int, Tuple[int, int, int]]:
        """严格控制颜色数的SVG颜色匹配"""
        
        print(f"\n开始色板匹配 (目标颜色数: {target_k})")
        print(f"输入: {len(set(quantized_mapping.values()))} 种量化颜色")
        
        final_mapping = {}
        quantized_indices = set(quantized_mapping.keys())
        
        # 分离皮肤和环境元素
        skin_elements_in_map = [(idx, quantized_mapping[idx]) for idx, _ in skin_elements if idx in quantized_indices]
        env_elements_in_map = [(idx, quantized_mapping[idx]) for idx, _ in env_elements if idx in quantized_indices]

        # 提取独特颜色
        skin_colors_to_match = list(set(color for _, color in skin_elements_in_map))
        env_colors_to_match = list(set(color for _, color in env_elements_in_map))
        
        print(f"皮肤区域: {len(skin_colors_to_match)} 种颜色, {len(skin_elements_in_map)} 个元素")
        print(f"环境区域: {len(env_colors_to_match)} 种颜色, {len(env_elements_in_map)} 个元素")

        # 执行匹配
        skin_map, palette_name = self.match_skin_colors(skin_colors_to_match, original_skin_colors)
        env_map = self.match_environment_colors(env_colors_to_match)

        print(f"选择的皮肤色板: {palette_name}")

        # 应用匹配结果
        for idx, quantized_color in quantized_mapping.items():
            is_skin = any(idx == skin_idx for skin_idx, _ in skin_elements)
            
            if is_skin and quantized_color in skin_map:
                final_mapping[idx] = skin_map[quantized_color]
            elif not is_skin and quantized_color in env_map:
                final_mapping[idx] = env_map[quantized_color]
            else:
                # 备用匹配
                final_mapping[idx] = quantized_color

        # 严格验证和限制颜色数
        unique_final_colors = list(set(final_mapping.values()))
        actual_colors = len(unique_final_colors)
        
        print(f"匹配后颜色数: {actual_colors}")
        
        if target_k and actual_colors > target_k:
            print(f"警告: 颜色数超标 ({actual_colors} > {target_k})，进行压缩...")
            final_mapping = self._compress_colors_to_target(final_mapping, target_k, skin_elements, env_elements)
            
            final_colors_after_compression = len(set(final_mapping.values()))
            print(f"压缩后颜色数: {final_colors_after_compression}")

        return final_mapping
    
    def _compress_colors_to_target(self, 
                                 color_mapping: Dict[int, Tuple[int, int, int]], 
                                 target_k: int,
                                 skin_elements: List[Tuple[int, Tuple[int, int, int]]],
                                 env_elements: List[Tuple[int, Tuple[int, int, int]]]) -> Dict[int, Tuple[int, int, int]]:
        """将颜色数压缩到目标数量"""
        
        current_colors = list(set(color_mapping.values()))
        if len(current_colors) <= target_k:
            return color_mapping
        
        # 分析颜色使用频率
        color_usage = {}
        skin_indices = set(idx for idx, _ in skin_elements)
        
        for idx, color in color_mapping.items():
            if color not in color_usage:
                color_usage[color] = {"count": 0, "is_skin": False, "elements": []}
            color_usage[color]["count"] += 1
            color_usage[color]["elements"].append(idx)
            if idx in skin_indices:
                color_usage[color]["is_skin"] = True
        
        # 保护重要颜色（皮肤颜色和高频颜色）
        protected_colors = []
        remaining_colors = []
        
        for color, usage in color_usage.items():
            if usage["is_skin"] or usage["count"] >= 3:  # 保护皮肤颜色和高频颜色
                protected_colors.append(color)
            else:
                remaining_colors.append((color, usage["count"]))
        
        # 按使用频率排序剩余颜色
        remaining_colors.sort(key=lambda x: x[1], reverse=True)
        
        # 选择要保留的颜色
        colors_to_keep = protected_colors[:]
        available_slots = target_k - len(protected_colors)
        
        for color, _ in remaining_colors[:available_slots]:
            colors_to_keep.append(color)
        
        # 为被删除的颜色找到替代颜色
        colors_to_remove = set(current_colors) - set(colors_to_keep)
        
        if colors_to_remove:
            # 构建替代映射
            replacement_map = {}
            colors_to_keep_lab = [rgb_to_lab(c) for c in colors_to_keep]
            
            for remove_color in colors_to_remove:
                remove_lab = rgb_to_lab(remove_color)
                distances = [delta_e_2000(remove_lab, keep_lab) for keep_lab in colors_to_keep_lab]
                best_replacement = colors_to_keep[np.argmin(distances)]
                replacement_map[remove_color] = best_replacement
            
            # 应用替代
            new_mapping = {}
            for idx, color in color_mapping.items():
                if color in replacement_map:
                    new_mapping[idx] = replacement_map[color]
                else:
                    new_mapping[idx] = color
            
            return new_mapping
        
        return color_mapping

# 保持向后兼容
SVGPaletteMatcher = RobustSVGPaletteMatcher