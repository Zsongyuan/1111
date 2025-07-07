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
from svg_region_mapper import SVGRegionMapper, FastSVGRegionMapper, SVGElement
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
            # 根据处理模式选择合适的映射器
            mode = getattr(svg_config, 'PROCESSING_MODE', 'fast')
            if mode == 'fast':
                print("使用快速区域映射器...")
                region_mapper = FastSVGRegionMapper(svg_parser, skin_mask)
            else:
                print("使用精确区域映射器...")
                region_mapper = SVGRegionMapper(svg_parser, skin_mask)
            
            skin_indices, env_indices = region_mapper.map_regions()

            # --- 修复核心：调整代码执行顺序 ---
            # 1. 在调用权重计算之前，先根据索引准备好完整的皮肤SVGElement对象列表
            skin_indices_set = set(skin_indices)
            all_skin_svg_elements = [e for e in elements if e.index in skin_indices_set]

            # 2. 将准备好的列表作为参数，传递给权重计算函数
            importance_weights = region_mapper.get_element_importance_weights(all_skin_svg_elements)
            # --- 修复结束 ---

            print(f"皮肤元素: {len(skin_indices)} 个")
            print(f"环境元素: {len(env_indices)} 个")
            
            # 6. 准备颜色数据 (这部分保持不变)
            skin_elements = [(idx, elements[idx].fill_color) for idx in skin_indices]
            env_elements = [(idx, elements[idx].fill_color) for idx in env_indices]
            
            # 7. 迭代优化颜色分配
            print("\n开始颜色量化...")
            best_result = self._iterative_color_optimization(
                skin_elements, env_elements, target_k, skin_ratio, 
                importance_weights, region_mapper
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
                                    region_mapper: SVGRegionMapper,
                                    max_iterations: int = 5) -> Dict[int, Tuple[int, int, int]]:
        """
        (已修复) 迭代优化颜色分配，并集成关键特征颜色保留机制。
        """
        reserved_color_mapping = {}
        quantization_skin_elements = []
        
        print("\n识别并保留关键特征颜色(眼睛、嘴唇)...")
        # 获取完整的SVGElement对象列表以进行分析
        skin_indices_set = {idx for idx, _ in skin_elements}
        all_skin_svg_elements = [e for e in region_mapper.svg_parser.elements if e.index in skin_indices_set]
        
        for idx, color in skin_elements:
            svg_element = next((e for e in all_skin_svg_elements if e.index == idx), None)
            # 调用新的、更精确的特征识别函数
            if svg_element and region_mapper._is_likely_facial_feature(svg_element, all_skin_svg_elements):
                reserved_color_mapping[idx] = color
            else:
                quantization_skin_elements.append((idx, color))

        num_reserved_colors = len(set(reserved_color_mapping.values()))
        print(f"已保留 {len(reserved_color_mapping)} 个元素, 共 {num_reserved_colors} 种关键颜色。")

        # --- 核心逻辑修复与加固 ---
        # 如果保留颜色过多，发出警告并自动禁用保留，以防止崩溃
        if num_reserved_colors >= target_k // 2 and num_reserved_colors > 5:
            print(f"警告：保留的颜色({num_reserved_colors})过多，已自动禁用保留机制以确保程序稳定。")
            reserved_color_mapping.clear()
            quantization_skin_elements = skin_elements
            num_reserved_colors = 0

        current_k = target_k
        best_result = {}
        # 将保留的颜色先放入最佳结果，确保它们始终存在
        best_result.update(reserved_color_mapping)
        best_diff = float('inf')

        for iteration in range(max_iterations):
            # 1. 确保量化目标数是有效的正数
            quantization_target_k = max(1, current_k - num_reserved_colors)
            
            # 2. 计算颜色分配
            k_skin, k_env = self._calculate_color_distribution(
                quantization_target_k, skin_ratio, len(quantization_skin_elements), len(env_elements)
            )
            print(f"\n迭代 {iteration + 1}: 量化目标(皮肤 {k_skin}, 环境 {k_env}) | 总目标 {current_k} | 已保留 {num_reserved_colors}")

            # 3. 对未保留的颜色执行量化
            color_mapping = self.quantizer.quantize_by_regions(
                quantization_skin_elements, env_elements, k_skin, k_env, 
                importance_weights, SATURATION_FACTOR
            )
            
            # 4. 将保留的颜色加回当前迭代的结果中
            color_mapping.update(reserved_color_mapping)
            
            unique_colors = len(set(color_mapping.values()))
            diff = abs(unique_colors - target_k)
            print(f"迭代 {iteration + 1}: 实际颜色数 = {unique_colors}")

            if diff < best_diff:
                best_diff = diff
                best_result = color_mapping.copy()
            
            if diff <= 2: # 如果误差在2以内，视为成功
                print(f"已达到目标颜色数，差异: {diff}")
                break
            
            # 5. 更新下一次迭代的总目标数
            current_k = self.color_strategy.suggest_adjustment(
                actual_colors=unique_colors,
                target_colors=target_k,
                current_k=current_k
            )
        
        # 6. 如果从未成功执行过，确保返回一个有效（但可能不理想）的结果
        if not best_result:
            # 合并环境和保留的颜色，确保至少有东西输出
            env_mapping = self.quantizer.quantize_by_regions([], env_elements, 1, max(1, target_k - num_reserved_colors), importance_weights, SATURATION_FACTOR)
            best_result.update(env_mapping)

        final_colors = len(set(best_result.values()))
        print(f"\n最终颜色数量: {final_colors}，目标颜色数: {target_k}")
            
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