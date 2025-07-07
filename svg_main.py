# svg_main.py
"""
主处理模块：整合所有功能实现SVG到SVG的数字油画转换
"""
import os
import numpy as np
import cv2
from typing import Tuple, Dict, List
import tempfile
import shutil

# 导入项目模块
import config
import getK
from api import get_skin_mask
from svg_parser import SVGParser
from svg_region_mapper import SVGRegionMapper, FastSVGRegionMapper
from svg_color_quantizer import RegionAwareSVGQuantizer
from svg_palette_matching import SVGPaletteMatcher
from svg_output import SVGOutput, BatchSVGProcessor
import svg_config

# 尝试导入颜色分配策略，如果失败则使用默认实现
try:
    from color_distribution_strategy import ColorDistributionStrategy
except ImportError:
    print("警告: 无法导入color_distribution_strategy，使用内置实现")
    # 内置的简单实现
    class ColorDistributionStrategy:
        def __init__(self, **kwargs):
            self.skin_ratio_threshold = kwargs.get('skin_ratio_threshold', 0.1)
            self.min_skin_colors = kwargs.get('min_skin_colors', 5)
            self.default_skin_weight = kwargs.get('default_skin_weight', 3)
            self.enhanced_skin_weight = kwargs.get('enhanced_skin_weight', 5)
            
        def calculate_distribution(self, target_k, skin_ratio, n_skin_elements, n_env_elements, **kwargs):
            weight_skin = self.enhanced_skin_weight if skin_ratio > self.skin_ratio_threshold else self.default_skin_weight
            weighted_skin = n_skin_elements * weight_skin
            weighted_env = n_env_elements
            total_weight = weighted_skin + weighted_env
            if total_weight == 0:
                return target_k // 2, target_k - target_k // 2
            k_skin = max(1, int(round(target_k * weighted_skin / total_weight)))
            k_env = target_k - k_skin
            if skin_ratio > self.skin_ratio_threshold and k_skin < self.min_skin_colors:
                k_skin = min(self.min_skin_colors, target_k - 1)
                k_env = target_k - k_skin
            k_skin = max(1, min(k_skin, target_k - 1))
            k_env = max(1, target_k - k_skin)
            return k_skin, k_env
            
        def suggest_adjustment(self, actual_colors, target_colors, current_k):
            diff = target_colors - actual_colors
            if abs(diff) <= 2:
                return current_k
            return current_k + diff

# 从配置文件加载常量
SKIN_RATIO_THRESHOLD = svg_config.COLOR_DISTRIBUTION_CONFIG["skin_ratio_threshold"]
MIN_SKIN_COLORS = svg_config.COLOR_DISTRIBUTION_CONFIG["min_skin_colors"]
DEFAULT_SKIN_WEIGHT = svg_config.COLOR_DISTRIBUTION_CONFIG["default_skin_weight"]
ENHANCED_SKIN_WEIGHT = svg_config.COLOR_DISTRIBUTION_CONFIG["enhanced_skin_weight"]
SATURATION_FACTOR = svg_config.QUANTIZATION_CONFIG["saturation_factor"]

class SVGDigitalPaintingProcessor:
    """SVG数字油画处理器"""
    
    def __init__(self, use_gpu: bool = True):
        self.use_gpu = use_gpu
        self.quantizer = RegionAwareSVGQuantizer(use_gpu)
        self.palette_matcher = SVGPaletteMatcher()
        self.batch_processor = BatchSVGProcessor()
        # 使用改进的颜色分配策略
        self.color_strategy = ColorDistributionStrategy(
            skin_ratio_threshold=SKIN_RATIO_THRESHOLD,
            min_skin_colors=MIN_SKIN_COLORS,
            default_skin_weight=DEFAULT_SKIN_WEIGHT,
            enhanced_skin_weight=ENHANCED_SKIN_WEIGHT
        )
        
    def process_file(self, svg_file_path: str, output_dir: str, dpi: int = 300):
        """
        处理单个SVG文件
        
        步骤:
        1. 提取目标颜色数
        2. 解析SVG文件
        3. 生成位图用于皮肤分割
        4. 获取皮肤mask
        5. 将mask映射到SVG元素
        6. 分区域进行颜色量化
        7. 色板匹配
        8. 输出结果
        """
        print(f"\n{'='*60}")
        print(f"开始处理: {svg_file_path}")
        print(f"{'='*60}")
        
        # 1. 提取目标颜色数
        target_k = getK.extract_k_value(svg_file_path)
        print(f"目标颜色数: {target_k}")
        
        # 2. 解析SVG文件
        print("\n解析SVG文件...")
        svg_parser = SVGParser(svg_file_path)
        elements = svg_parser.parse()
        print(f"找到 {len(elements)} 个path元素")
        
        # 3. 生成位图用于皮肤分割
        print("\n生成位图用于皮肤分割...")
        bitmap = svg_parser.render_to_bitmap(dpi=dpi)
        
        # 保存临时位图文件
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
            temp_bitmap_path = tmp_file.name
            cv2.imwrite(temp_bitmap_path, cv2.cvtColor(bitmap, cv2.COLOR_RGB2BGR))
        
        try:
            # 4. 获取皮肤mask
            print("\n调用皮肤分割API...")
            skin_mask = get_skin_mask(temp_bitmap_path)
            
            # 计算皮肤区域占比
            skin_ratio = np.sum(skin_mask > 0) / (skin_mask.shape[0] * skin_mask.shape[1])
            print(f"皮肤区域占比: {skin_ratio:.2%}")
            
            # 5. 将mask映射到SVG元素
            print("\n映射皮肤区域到SVG元素...")
            region_mapper = SVGRegionMapper(svg_parser, skin_mask)
            skin_indices, env_indices = region_mapper.map_regions()
            importance_weights = region_mapper.get_element_importance_weights()
            
            print(f"皮肤元素: {len(skin_indices)} 个")
            print(f"环境元素: {len(env_indices)} 个")
            
            # 6. 准备颜色数据
            skin_elements = [(idx, elements[idx].fill_color) for idx in skin_indices]
            env_elements = [(idx, elements[idx].fill_color) for idx in env_indices]
            
            # 7. 迭代优化颜色分配
            print("\n开始颜色量化...")
            best_result = self._iterative_color_optimization(
                skin_elements, env_elements, target_k, skin_ratio, importance_weights
            )
            
            # 8. 色板匹配
            print("\n进行色板匹配...")
            final_color_mapping = self._apply_palette_matching(
                best_result, skin_elements, env_elements
            )
            
            # 9. 输出结果
            print("\n生成输出文件...")
            self._save_results(svg_parser, final_color_mapping, svg_file_path, output_dir, dpi)
            
        finally:
            # 清理临时文件
            if os.path.exists(temp_bitmap_path):
                os.remove(temp_bitmap_path)
    
    def _iterative_color_optimization(self,
                                    skin_elements: List[Tuple[int, Tuple[int, int, int]]],
                                    env_elements: List[Tuple[int, Tuple[int, int, int]]],
                                    target_k: int,
                                    skin_ratio: float,
                                    importance_weights: Dict[int, float],
                                    max_iterations: int = 5) -> Dict[int, Tuple[int, int, int]]:
        """
        迭代优化颜色分配直到接近目标颜色数
        参考位图版本的迭代策略
        """
        current_k = target_k
        best_result = None
        best_diff = float('inf')
        
        for iteration in range(max_iterations):
            # 计算颜色分配
            k_skin, k_env = self._calculate_color_distribution(
                current_k, skin_ratio, len(skin_elements), len(env_elements)
            )
            
            print(f"\n迭代 {iteration + 1}: 分配颜色 - 皮肤 {k_skin} 色，环境 {k_env} 色 (当前总目标 {current_k} 色)")
            
            # 执行量化
            color_mapping = self.quantizer.quantize_by_regions(
                skin_elements, env_elements, k_skin, k_env, 
                importance_weights, SATURATION_FACTOR
            )
            
            # 统计实际颜色数
            unique_colors = len(set(color_mapping.values()))
            diff = abs(unique_colors - target_k)
            
            print(f"迭代 {iteration + 1}: 实际颜色数 = {unique_colors}")
            
            # 保存最佳结果
            if diff < best_diff:
                best_diff = diff
                best_result = color_mapping
            
            # 如果达到目标，退出
            if diff <= 2:  # 容差范围
                print(f"达到目标颜色数，差异: {diff}")
                break
            
            # 使用策略类建议下一次的调整
            current_k = self.color_strategy.suggest_adjustment(
                actual_colors=unique_colors,
                target_colors=target_k,
                current_k=current_k
            )
        
        if best_result is None:
            # 如果没有找到好的结果，使用最后一次的结果
            best_result = color_mapping
            
        # 打印最终统计
        final_colors = len(set(best_result.values()))
        print(f"\n最终颜色数量: {final_colors}，目标颜色数: {target_k}")
        if abs(final_colors - target_k) > 2:
            print("警告：最终颜色数量与目标颜色数差异较大！")
            
        return best_result
    
    def _calculate_color_distribution(self, 
                                    total_k: int, 
                                    skin_ratio: float,
                                    n_skin_elements: int,
                                    n_env_elements: int) -> Tuple[int, int]:
        """
        计算皮肤和环境区域的颜色分配
        使用改进的颜色分配策略
        """
        return self.color_strategy.calculate_distribution(
            target_k=total_k,
            skin_ratio=skin_ratio,
            n_skin_elements=n_skin_elements,
            n_env_elements=n_env_elements
        )
    
    def _apply_palette_matching(self,
                               quantized_mapping: Dict[int, Tuple[int, int, int]],
                               skin_elements: List[Tuple[int, Tuple[int, int, int]]],
                               env_elements: List[Tuple[int, Tuple[int, int, int]]]) -> Dict[int, Tuple[int, int, int]]:
        """应用色板匹配"""
        # 根据量化结果分组元素
        skin_color_groups = self._group_elements_by_color(quantized_mapping, skin_elements)
        env_color_groups = self._group_elements_by_color(quantized_mapping, env_elements)
        
        # 执行色板匹配
        final_mapping = self.palette_matcher.match_svg_colors(
            skin_color_groups, env_color_groups, skin_elements
        )
        
        return final_mapping
    
    def _group_elements_by_color(self,
                                color_mapping: Dict[int, Tuple[int, int, int]],
                                elements: List[Tuple[int, Tuple[int, int, int]]]) -> List[Tuple[List[int], Tuple[int, int, int]]]:
        """按颜色分组元素"""
        color_to_indices = {}
        
        for idx, _ in elements:
            if idx in color_mapping:
                color = color_mapping[idx]
                if color not in color_to_indices:
                    color_to_indices[color] = []
                color_to_indices[color].append(idx)
        
        return [(indices, color) for color, indices in color_to_indices.items()]
    
    def _save_results(self,
                     svg_parser: SVGParser,
                     color_mapping: Dict[int, Tuple[int, int, int]],
                     original_path: str,
                     output_dir: str,
                     dpi: int):
        """保存处理结果"""
        base_name = os.path.splitext(os.path.basename(original_path))[0]
        
        # 验证颜色映射
        print("\n验证颜色映射...")
        for idx, color in color_mapping.items():
            if not isinstance(color, tuple) or len(color) != 3:
                print(f"警告: 元素 {idx} 的颜色格式错误: {color}")
            elif not all(isinstance(c, (int, np.integer)) for c in color):
                # 尝试修复
                fixed_color = tuple(int(c) for c in color)
                color_mapping[idx] = fixed_color
                print(f"修复: 元素 {idx} 的颜色从 {color} 转换为 {fixed_color}")
        
        # 创建SVG输出器
        svg_output = SVGOutput(svg_parser)
        
        # 更新颜色
        svg_output.update_svg_colors(color_mapping)
        
        # 保存SVG
        svg_path = os.path.join(output_dir, f"{base_name}_digital_painting.svg")
        svg_output.save_svg(svg_path)
        
        # 渲染并保存位图
        bitmap_path = os.path.join(output_dir, f"{base_name}_digital_painting.png")
        svg_output.render_to_bitmap(bitmap_path, dpi)
        
        # 生成对比图
        comparison_path = os.path.join(output_dir, f"{base_name}_comparison.png")
        self.batch_processor.create_comparison_image(
            original_path, svg_path, comparison_path, dpi
        )
        
        # 生成色板图
        palette_path = os.path.join(output_dir, f"{base_name}_palette.png")
        self.batch_processor.generate_color_palette_image(color_mapping, palette_path)
        
        # 打印统计信息
        unique_colors = len(set(color_mapping.values()))
        print(f"\n处理完成!")
        print(f"最终颜色数: {unique_colors}")
        print(f"输出文件:")
        print(f"  - SVG: {svg_path}")
        print(f"  - PNG: {bitmap_path}")
        print(f"  - 对比图: {comparison_path}")
        print(f"  - 色板: {palette_path}")
    
    def process_folder(self, input_dir: str, output_dir: str, dpi: int = 300):
        """批量处理文件夹中的所有SVG文件"""
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)
        
        # 查找所有SVG文件
        svg_files = [
            os.path.join(input_dir, f) 
            for f in os.listdir(input_dir) 
            if f.lower().endswith('.svg')
        ]
        
        if not svg_files:
            print("未找到SVG文件")
            return
        
        print(f"找到 {len(svg_files)} 个SVG文件")
        
        # 处理每个文件
        for i, svg_file in enumerate(svg_files, 1):
            print(f"\n\n处理进度: {i}/{len(svg_files)}")
            try:
                self.process_file(svg_file, output_dir, dpi)
            except Exception as e:
                print(f"处理失败: {e}")
                import traceback
                traceback.print_exc()

def main():
    """主函数"""
    # 从配置获取目录
    input_dir = config.CONFIG["INPUT_DIR"]
    output_dir = config.CONFIG["OUTPUT_DIR"]
    
    # 创建处理器
    processor = SVGDigitalPaintingProcessor(use_gpu=True)
    
    # 批量处理
    processor.process_folder(input_dir, output_dir, dpi=300)

if __name__ == '__main__':
    main()