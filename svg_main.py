# svg_main.py
"""
主处理模块：整合所有功能实现SVG到SVG的数字油画转换
"""
import os
import numpy as np
import cv2
from typing import Tuple, Dict, List
import tempfile

# 导入项目模块
import config, getK, svg_config
from api import get_skin_mask
from svg_parser import SVGParser
from svg_region_mapper import SVGRegionMapper, FastSVGRegionMapper
from svg_color_quantizer import RegionAwareSVGQuantizer
from svg_palette_matching import SVGPaletteMatcher
from svg_output import SVGOutput

try:
    from color_distribution_strategy import ColorDistributionStrategy
except ImportError:
    # ... 内置的 ColorDistributionStrategy 无变化 ...
    print("警告: 无法导入color_distribution_strategy，使用内置实现")
    class ColorDistributionStrategy:
        def __init__(self, **kwargs): self.__dict__.update(kwargs)
        def calculate_distribution(self, target_k, skin_ratio, n_skin_elements, n_env_elements, **kwargs):
            weight_skin = self.enhanced_skin_weight if skin_ratio > self.skin_ratio_threshold else self.default_skin_weight
            weighted_skin = n_skin_elements * weight_skin; weighted_env = n_env_elements
            total_weight = weighted_skin + weighted_env
            if total_weight == 0: return target_k // 2, target_k - target_k // 2
            k_skin = int(round(target_k*weighted_skin/total_weight))
            k_env = target_k - k_skin
            if k_skin < 1 and k_env > 1: k_skin=1; k_env=target_k-1
            if k_env < 1 and k_skin > 1: k_env=1; k_skin=target_k-1
            return k_skin, k_env
        def suggest_adjustment(self, actual, target, current):
            if abs(actual - target) <= 1: return current
            return current + (target - actual)
        def suggest_adjustment(self, actual, target, current): return current

SATURATION_FACTOR = svg_config.QUANTIZATION_CONFIG["saturation_factor"]

class SVGDigitalPaintingProcessor:
    """SVG数字油画处理器"""
    
    def __init__(self, use_gpu: bool = True):
        self.quantizer = RegionAwareSVGQuantizer(use_gpu)
        self.palette_matcher = SVGPaletteMatcher()
        self.color_strategy = ColorDistributionStrategy(**svg_config.COLOR_DISTRIBUTION_CONFIG)

    def process_file(self, svg_file_path: str, output_dir: str, dpi: int = 300):
        print(f"\n{'='*60}\n开始处理: {os.path.basename(svg_file_path)}\n{'='*60}")
        target_k = getK.extract_k_value(svg_file_path)
        print(f"目标颜色数: {target_k}")
        
        svg_parser = SVGParser(svg_file_path)
        elements = svg_parser.parse()
        
        # 渲染位图，用于皮肤分割和颜色分配
        bitmap = svg_parser.render_to_bitmap(dpi=dpi)
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
            temp_bitmap_path = tmp_file.name
            # imwrite需要BGR格式
            cv2.imwrite(temp_bitmap_path, cv2.cvtColor(bitmap, cv2.COLOR_RGB2BGR))
        
        try:
            print("\n调用皮肤分割API...")
            skin_mask = get_skin_mask(temp_bitmap_path)
            skin_ratio = np.sum(skin_mask > 0) / (skin_mask.size or 1)
            
            mode = getattr(svg_config, 'PROCESSING_MODE', 'fast')
            region_mapper = (FastSVGRegionMapper if mode == 'fast' else SVGRegionMapper)(svg_parser, skin_mask)
            skin_indices, env_indices, eye_indices, mouth_indices = region_mapper.map_regions()

            facial_indices = sorted(list(set(skin_indices + eye_indices + mouth_indices)))
            
            skin_elements = [(i, elements[i].fill_color) for i in skin_indices]
            env_elements = [(i, elements[i].fill_color) for i in env_indices]
            facial_elements = [(i, elements[i].fill_color) for i in facial_indices]
            
            importance_weights = region_mapper.get_element_importance_weights(skin_indices, eye_indices, mouth_indices)

            print("\n开始迭代式颜色量化...")
            # 【核心修复】将 bitmap 和 skin_mask 传递给迭代量化函数
            quantized_mapping = self._quantize_regions_iteratively(
                facial_elements, env_elements, target_k, skin_ratio, importance_weights, bitmap, skin_mask
            )
            
            print("\n进行色板匹配...")
            final_color_mapping = self.palette_matcher.match_svg_colors(
                quantized_mapping, skin_elements, env_elements
            )
            
            self._save_results(svg_parser, final_color_mapping, svg_file_path, output_dir)
            
        finally:
            if os.path.exists(temp_bitmap_path):
                os.remove(temp_bitmap_path)


    def _quantize_regions_iteratively(self,
                                    facial_elements: List,
                                    env_elements: List,
                                    target_k: int,
                                    skin_ratio: float,
                                    importance_weights: Dict[int, float],
                                    bitmap: np.ndarray,         # <-- 接收 bitmap
                                    skin_mask: np.ndarray,        # <-- 接收 skin_mask
                                    max_iterations: int = 5) -> Dict[int, Tuple[int, int, int]]:
        current_k = target_k
        best_result, best_diff = {}, float('inf')

        for iteration in range(max_iterations):
            # 【核心修复】使用正确的参数调用新的颜色分配策略
            k_facial, k_env = self.color_strategy.calculate_distribution(
                current_k, skin_ratio, bitmap, skin_mask
            )
            print(f"迭代 {iteration+1}: 量化目标(面部 {k_facial}, 环境 {k_env}) | 总目标 {current_k}")

            color_mapping = self.quantizer.quantize_by_regions(
                facial_elements, env_elements, k_facial, k_env, 
                importance_weights, SATURATION_FACTOR
            )
            
            unique_colors = len(set(color_mapping.values()))
            if unique_colors == 0 and current_k > 0:
                 print("警告：量化未产生颜色，提前中止。"); break

            diff = abs(unique_colors - current_k)
            print(f"迭代 {iteration+1}: 实际颜色数 = {unique_colors}")

            if diff < best_diff:
                best_diff, best_result = diff, color_mapping.copy()
            if diff <= 1: break
            
            current_k = self.color_strategy.suggest_adjustment(unique_colors, current_k, current_k)
        
        final_colors = len(set(best_result.values()))
        print(f"\n量化完成，最终颜色数: {final_colors}")
        return best_result

    def _save_results(self, svg_parser, color_mapping, original_path, output_dir):
        base_name = os.path.basename(original_path)
        output_svg_path = os.path.join(output_dir, base_name)
        svg_output = SVGOutput(svg_parser)
        try:
            svg_output.update_svg_colors(color_mapping)
            svg_output.save_svg(output_svg_path)
            final_colors = len(set(color_mapping.values()))
            print(f"  > 处理完成! 最终颜色数: {final_colors}. 输出文件: {output_svg_path}")
        except Exception as e:
            print(f"  > 严重错误：保存最终SVG文件失败: {e}")
    
    def process_folder(self, input_dir: str, output_dir: str, dpi: int = 300):
        os.makedirs(output_dir, exist_ok=True)
        svg_files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.lower().endswith('.svg')]
        if not svg_files: print("未找到SVG文件"); return
        
        print(f"找到 {len(svg_files)} 个SVG文件")
        for i, svg_file in enumerate(svg_files, 1):
            try: self.process_file(svg_file, output_dir, dpi)
            except Exception as e:
                print(f"处理文件 {os.path.basename(svg_file)} 时发生严重错误: {e}"); import traceback; traceback.print_exc()

def main():
    SVGDigitalPaintingProcessor().process_folder(config.CONFIG["INPUT_DIR"], config.CONFIG["OUTPUT_DIR"])

if __name__ == '__main__':
    main()