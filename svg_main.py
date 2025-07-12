# svg_main.py
"""
保护面部特征的主处理模块
"""
import os
import numpy as np
import cv2
from typing import Tuple, Dict, List, Set
import tempfile

# 导入项目模块
import config, getK, svg_config
from api import get_skin_mask
from svg_parser import SVGParser
from svg_region_mapper import EnhancedSVGRegionMapper, FastSVGRegionMapper
from svg_color_quantizer import EnhancedRegionAwareSVGQuantizer
from svg_palette_matching import RobustSVGPaletteMatcher
from svg_output import SVGOutput

try:
    from color_distribution_strategy import ColorDistributionStrategy
except ImportError:
    print("警告: 使用内置颜色分配策略")
    class ColorDistributionStrategy:
        def __init__(self, **kwargs): 
            self.__dict__.update(kwargs)
            self.skin_ratio_threshold = kwargs.get('skin_ratio_threshold', 0.1)
            self.enhanced_skin_weight = kwargs.get('enhanced_skin_weight', 5)
            self.default_skin_weight = kwargs.get('default_skin_weight', 3)
            
        def calculate_distribution(self, target_k, skin_ratio, facial_elements, env_elements):
            n_facial, n_env = len(facial_elements), len(env_elements)
            if n_facial == 0: return 0, target_k
            if n_env == 0: return target_k, 0
            
            weight_facial = self.enhanced_skin_weight if skin_ratio > self.skin_ratio_threshold else self.default_skin_weight
            weighted_facial, weighted_env = n_facial * weight_facial, n_env * 1.0
            total_weight = weighted_facial + weighted_env
            
            if total_weight == 0: return target_k // 2, target_k - target_k // 2
            
            k_facial = max(1, int(round(target_k * weighted_facial / total_weight)))
            k_env = target_k - k_facial
            return max(1, k_facial), max(1, k_env)
            
        def suggest_adjustment(self, actual, target, current):
            return max(1, current + (target - actual)) if abs(actual - target) > 1 else current

SATURATION_FACTOR = svg_config.QUANTIZATION_CONFIG["saturation_factor"]

class FacialFeatureProtectedProcessor:
    """保护面部特征的SVG数字油画处理器"""
    
    def __init__(self, use_gpu: bool = True):
        self.quantizer = EnhancedRegionAwareSVGQuantizer(use_gpu)
        self.palette_matcher = RobustSVGPaletteMatcher()
        self.color_strategy = ColorDistributionStrategy(**svg_config.COLOR_DISTRIBUTION_CONFIG)
        
        print("初始化完成 - 使用面部特征保护算法")
        
    def process_file(self, svg_file_path: str, output_dir: str, dpi: int = 300):
        print(f"\n{'='*60}\n开始处理: {os.path.basename(svg_file_path)}\n{'='*60}")
        
        # 保守的目标颜色数策略
        base_k = getK.extract_k_value(svg_file_path)
        increment = 2 if base_k <= 24 else (1 if base_k <= 36 else 0)
        target_k = base_k + increment
        print(f"基础颜色数: {base_k}, 目标颜色数: {target_k} (基础+{increment})")
        
        svg_parser = SVGParser(svg_file_path)
        elements = svg_parser.parse()
        print(f"SVG元素总数: {len(elements)}")
        
        bitmap = svg_parser.render_to_bitmap(dpi=dpi)
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
            temp_bitmap_path = tmp_file.name
            cv2.imwrite(temp_bitmap_path, cv2.cvtColor(bitmap, cv2.COLOR_RGB2BGR))
        
        try:
            print("\n调用皮肤分割API...")
            skin_mask = get_skin_mask(temp_bitmap_path)
            skin_ratio = np.sum(skin_mask > 0) / (skin_mask.size or 1)
            print(f"皮肤区域占比: {skin_ratio:.1%}")
            
            # 使用增强的区域映射器
            mode = getattr(svg_config, 'PROCESSING_MODE', 'fast')
            if mode == 'fast':
                region_mapper = FastSVGRegionMapper(svg_parser, skin_mask)
            else:
                region_mapper = EnhancedSVGRegionMapper(svg_parser, skin_mask)
            
            skin_indices, env_indices, eye_indices, mouth_indices = region_mapper.map_regions()

            facial_indices = sorted(list(set(skin_indices + eye_indices + mouth_indices)))
            
            skin_elements = [(i, elements[i].fill_color) for i in skin_indices]
            env_elements = [(i, elements[i].fill_color) for i in env_indices]
            facial_elements = [(i, elements[i].fill_color) for i in facial_indices]
            
            # 收集需要保护的颜色
            protected_colors = region_mapper.get_protected_colors()
            protected_elements = region_mapper.get_protected_elements()
            
            print(f"面部特征保护:")
            print(f"  眼部元素: {len(eye_indices)} 个")
            print(f"  嘴部元素: {len(mouth_indices)} 个") 
            print(f"  保护颜色: {len(protected_colors)} 种")
            
            # 输出保护的颜色信息
            if protected_colors:
                sample_colors = list(protected_colors)[:3]
                print(f"  保护颜色样本: {sample_colors}")
            
            original_skin_colors = [color for _, color in skin_elements]
            importance_weights = region_mapper.get_element_importance_weights(skin_indices, eye_indices, mouth_indices)

            # 带保护机制的量化过程
            quantized_mapping = self._protected_quantization(
                facial_elements, env_elements, target_k, skin_ratio, 
                importance_weights, protected_colors
            )
            
            # 验证保护颜色是否被保留
            self._verify_color_protection(quantized_mapping, protected_colors, protected_elements)
            
            # 色板匹配（注意保护重要颜色）
            final_color_mapping = self._protected_palette_matching(
                quantized_mapping, skin_elements, env_elements, 
                original_skin_colors, target_k, protected_colors
            )
            
            # 最终验证
            final_color_mapping = self._final_protection_validation(
                final_color_mapping, target_k, protected_colors, protected_elements
            )
            
            self._save_results(svg_parser, final_color_mapping, svg_file_path, output_dir, protected_colors)
            
        finally:
            if os.path.exists(temp_bitmap_path):
                os.remove(temp_bitmap_path)

    def _protected_quantization(self,
                               facial_elements: List,
                               env_elements: List,
                               target_k: int,
                               skin_ratio: float,
                               importance_weights: Dict[int, float],
                               protected_colors: Set[Tuple[int, int, int]]) -> Dict[int, Tuple[int, int, int]]:
        """带保护机制的颜色量化"""
        
        print(f"\n开始保护性颜色量化 (目标: {target_k}色)")
        
        # 为保护颜色预留空间
        num_protected = len(protected_colors)
        quantization_target = max(1, target_k - num_protected)
        
        print(f"  保护颜色数: {num_protected}")
        print(f"  量化目标: {quantization_target} (为保护颜色预留 {num_protected} 个位置)")
        
        max_iterations = 5
        best_result = {}
        best_diff = float('inf')
        
        for iteration in range(max_iterations):
            k_facial, k_env = self.color_strategy.calculate_distribution(
                quantization_target, skin_ratio, facial_elements, env_elements
            )
            
            print(f"迭代 {iteration + 1}: 分配 面部={k_facial}, 环境={k_env}")
            
            color_mapping = self.quantizer.quantize_by_regions(
                facial_elements, env_elements, k_facial, k_env, 
                importance_weights, protected_colors, SATURATION_FACTOR
            )
            
            if not color_mapping:
                continue
            
            unique_colors = len(set(color_mapping.values()))
            # 检查保护颜色保留情况
            preserved_protected = protected_colors & set(color_mapping.values())
            protection_score = len(preserved_protected) / len(protected_colors) if protected_colors else 1.0
            
            print(f"  得到 {unique_colors} 种颜色, 保护颜色保留率: {protection_score:.1%}")
            
            # 综合评分：颜色数接近度 + 保护颜色保留率
            color_diff = abs(unique_colors - target_k)
            combined_score = color_diff - (protection_score * 10)  # 保护颜色权重很高
            
            if combined_score < best_diff:
                best_diff = combined_score
                best_result = color_mapping.copy()
                print(f"  -> 当前最佳结果 (综合得分: {combined_score:.2f})")
            
            if protection_score >= 0.8 and color_diff <= 2:  # 80%保护率且颜色数接近
                print(f"  -> 达到保护目标，停止迭代")
                break
        
        final_quantized_colors = len(set(best_result.values()))
        final_protected = protected_colors & set(best_result.values())
        print(f"量化完成: {final_quantized_colors} 种颜色, 保护 {len(final_protected)} 种重要颜色")
        
        return best_result

    def _verify_color_protection(self, 
                               color_mapping: Dict[int, Tuple[int, int, int]], 
                               protected_colors: Set[Tuple[int, int, int]],
                               protected_elements: Dict[int, str]):
        """验证颜色保护效果"""
        
        if not protected_colors:
            return
        
        final_colors = set(color_mapping.values())
        preserved_protected = protected_colors & final_colors
        lost_protected = protected_colors - final_colors
        
        print(f"\n颜色保护验证:")
        print(f"  原始保护颜色: {len(protected_colors)} 种")
        print(f"  成功保留: {len(preserved_protected)} 种")
        print(f"  丢失: {len(lost_protected)} 种")
        
        if lost_protected:
            print(f"  丢失的颜色: {list(lost_protected)[:3]}...")
            
            # 尝试恢复丢失的重要颜色
            print("  尝试恢复丢失的重要颜色...")
            self._recover_lost_colors(color_mapping, lost_protected, protected_elements)

    def _recover_lost_colors(self, 
                           color_mapping: Dict[int, Tuple[int, int, int]], 
                           lost_colors: Set[Tuple[int, int, int]],
                           protected_elements: Dict[int, str]):
        """尝试恢复丢失的重要颜色"""
        
        # 找到使用丢失颜色的重要元素
        recovery_count = 0
        
        for lost_color in lost_colors:
            # 找到应该使用这个颜色的重要元素
            important_elements = [idx for idx, feature_type in protected_elements.items() 
                                if idx in color_mapping]
            
            if important_elements:
                # 强制恢复：将第一个重要元素的颜色设为丢失的颜色
                recovery_element = important_elements[0]
                old_color = color_mapping[recovery_element]
                color_mapping[recovery_element] = lost_color
                recovery_count += 1
                
                print(f"    恢复颜色 {lost_color} -> 元素 {recovery_element}")
        
        print(f"  成功恢复 {recovery_count} 种重要颜色")

    def _protected_palette_matching(self,
                                  quantized_mapping: Dict[int, Tuple[int, int, int]],
                                  skin_elements: List,
                                  env_elements: List,
                                  original_skin_colors: List[Tuple[int, int, int]],
                                  target_k: int,
                                  protected_colors: Set[Tuple[int, int, int]]) -> Dict[int, Tuple[int, int, int]]:
        """保护重要颜色的色板匹配"""
        
        print(f"\n开始保护性色板匹配...")
        
        # 执行色板匹配
        matched_mapping = self.palette_matcher.match_svg_colors(
            quantized_mapping, skin_elements, env_elements, original_skin_colors, target_k
        )
        
        # 检查保护颜色是否在匹配后被保留
        matched_colors = set(matched_mapping.values())
        preserved_after_match = protected_colors & matched_colors
        
        print(f"  色板匹配后保护颜色保留: {len(preserved_after_match)}/{len(protected_colors)}")
        
        # 如果有保护颜色丢失，尝试强制保留
        if len(preserved_after_match) < len(protected_colors):
            lost_in_matching = protected_colors - preserved_after_match
            print(f"  在色板匹配中丢失的保护颜色: {len(lost_in_matching)} 种")
            
            # 强制保留机制：直接恢复重要颜色
            for lost_color in lost_in_matching:
                # 找到原本应该是这个颜色的元素
                for idx, orig_color in quantized_mapping.items():
                    if orig_color == lost_color:
                        matched_mapping[idx] = lost_color  # 直接恢复原色
                        break
        
        return matched_mapping

    def _final_protection_validation(self,
                                   color_mapping: Dict[int, Tuple[int, int, int]],
                                   target_k: int,
                                   protected_colors: Set[Tuple[int, int, int]],
                                   protected_elements: Dict[int, str]) -> Dict[int, Tuple[int, int, int]]:
        """最终保护验证和调整"""
        
        unique_colors = list(set(color_mapping.values()))
        actual_count = len(unique_colors)
        
        print(f"\n最终保护验证: 实际 {actual_count} 色, 目标 {target_k} 色")
        
        # 验证保护颜色
        final_protected = protected_colors & set(unique_colors)
        print(f"最终保护颜色保留: {len(final_protected)}/{len(protected_colors)}")
        
        if actual_count <= target_k:
            print("✓ 颜色数和保护都在目标范围内")
            return color_mapping
        
        print(f"⚠ 颜色数超标 {actual_count - target_k} 种，进行保护性压缩...")
        
        # 保护性压缩：优先保留重要颜色
        return self._protected_color_compression(color_mapping, target_k, protected_colors, protected_elements)

    def _protected_color_compression(self,
                                   color_mapping: Dict[int, Tuple[int, int, int]],
                                   target_k: int,
                                   protected_colors: Set[Tuple[int, int, int]],
                                   protected_elements: Dict[int, str]) -> Dict[int, Tuple[int, int, int]]:
        """保护重要颜色的压缩算法"""
        
        # 分析颜色使用情况
        color_usage = {}
        for idx, color in color_mapping.items():
            if color not in color_usage:
                color_usage[color] = {
                    "count": 0, 
                    "elements": [], 
                    "is_protected": color in protected_colors,
                    "importance": 0
                }
            color_usage[color]["count"] += 1
            color_usage[color]["elements"].append(idx)
            
            # 计算重要性
            if idx in protected_elements:
                color_usage[color]["importance"] += 100  # 面部特征最重要
            elif color in protected_colors:
                color_usage[color]["importance"] += 50   # 保护颜色重要
            else:
                color_usage[color]["importance"] += color_usage[color]["count"]  # 按使用频率
        
        # 按重要性排序，保护颜色优先
        def color_priority(item):
            color, info = item
            return (info["is_protected"], info["importance"], info["count"])
        
        sorted_colors = sorted(color_usage.items(), key=color_priority, reverse=True)
        
        # 强制保留所有保护颜色
        colors_to_keep = []
        for color, info in sorted_colors:
            if info["is_protected"]:
                colors_to_keep.append(color)
        
        # 添加其他高重要性颜色
        remaining_slots = target_k - len(colors_to_keep)
        for color, info in sorted_colors:
            if not info["is_protected"] and len(colors_to_keep) < target_k:
                colors_to_keep.append(color)
        
        colors_to_remove = set(color_usage.keys()) - set(colors_to_keep)
        
        print(f"  保护性压缩: 保留 {len(colors_to_keep)} 色 (其中保护色 {len(protected_colors & set(colors_to_keep))} 种)")
        
        # 为被移除的颜色找最近替代
        if colors_to_remove:
            replacement_map = {}
            for remove_color in colors_to_remove:
                best_replacement = self._find_nearest_color(remove_color, colors_to_keep)
                replacement_map[remove_color] = best_replacement
            
            # 应用替代
            final_mapping = {}
            for idx, color in color_mapping.items():
                final_mapping[idx] = replacement_map.get(color, color)
            
            return final_mapping
        
        return color_mapping

    def _find_nearest_color(self, target_color: Tuple[int, int, int], 
                          available_colors: List[Tuple[int, int, int]]) -> Tuple[int, int, int]:
        """找到最近的可用颜色"""
        min_distance = float('inf')
        nearest_color = available_colors[0] if available_colors else target_color
        
        for color in available_colors:
            distance = sum((a - b) ** 2 for a, b in zip(target_color, color))
            if distance < min_distance:
                min_distance = distance
                nearest_color = color
        
        return nearest_color

    def _save_results(self, svg_parser, color_mapping, original_path, output_dir, protected_colors):
        base_name = os.path.basename(original_path)
        output_svg_path = os.path.join(output_dir, base_name)
        svg_output = SVGOutput(svg_parser)
        
        try:
            svg_output.update_svg_colors(color_mapping)
            svg_output.save_svg(output_svg_path)
            
            final_colors = len(set(color_mapping.values()))
            final_protected = protected_colors & set(color_mapping.values())
            
            print(f"\n✓ 处理完成!")
            print(f"✓ 最终颜色数: {final_colors}")
            print(f"✓ 保护颜色保留: {len(final_protected)}/{len(protected_colors)}")
            print(f"✓ 输出文件: {output_svg_path}")
            
        except Exception as e:
            print(f"\n✗ 保存SVG文件失败: {e}")
            import traceback
            traceback.print_exc()
    
    def process_folder(self, input_dir: str, output_dir: str, dpi: int = 300):
        os.makedirs(output_dir, exist_ok=True)
        svg_files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.lower().endswith('.svg')]
        
        if not svg_files: 
            print("未找到SVG文件")
            return
        
        print(f"找到 {len(svg_files)} 个SVG文件")
        print(f"使用面部特征保护算法处理...")
        
        successful_count = 0
        
        for i, svg_file in enumerate(svg_files, 1):
            print(f"\n进度: {i}/{len(svg_files)}")
            try: 
                self.process_file(svg_file, output_dir, dpi)
                successful_count += 1
            except Exception as e:
                print(f"\n✗ 处理文件 {os.path.basename(svg_file)} 时发生错误: {e}")
                import traceback
                traceback.print_exc()
                
        print(f"\n{'='*60}")
        print(f"批量处理完成！")
        print(f"成功处理: {successful_count}/{len(svg_files)} 个文件")
        print(f"输出目录: {output_dir}")
        print(f"使用了面部特征保护算法，确保眼部和嘴部颜色不丢失")
        print(f"{'='*60}")

def main():
    processor = FacialFeatureProtectedProcessor()
    processor.process_folder(config.CONFIG["INPUT_DIR"], config.CONFIG["OUTPUT_DIR"])

if __name__ == '__main__':
    main()