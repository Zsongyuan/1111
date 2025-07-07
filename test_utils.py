# test_utils.py
"""
测试和调试工具模块
"""
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict
import time

from svg_parser import SVGParser
from svg_region_mapper import SVGRegionMapper
from svg_color_quantizer import SVGColorQuantizer
from api import get_skin_mask

class SVGDebugger:
    """SVG处理调试工具"""
    
    def __init__(self, output_dir: str = "debug_output"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
    def visualize_svg_elements(self, svg_file: str):
        """可视化SVG元素"""
        parser = SVGParser(svg_file)
        elements = parser.parse()
        
        print(f"\nSVG文件: {svg_file}")
        print(f"画布尺寸: {parser.width} x {parser.height}")
        print(f"元素总数: {len(elements)}")
        
        # 统计颜色
        color_counts = {}
        for elem in elements:
            color = elem.fill_color
            if color not in color_counts:
                color_counts[color] = 0
            color_counts[color] += 1
        
        print(f"\n颜色统计 (共{len(color_counts)}种颜色):")
        for color, count in sorted(color_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
            print(f"  RGB{color}: {count}个元素")
        
        # 生成颜色分布图
        self._plot_color_distribution(color_counts)
        
    def test_skin_segmentation(self, svg_file: str, save_intermediate: bool = True):
        """测试皮肤分割功能"""
        print(f"\n测试皮肤分割: {svg_file}")
        
        # 解析SVG
        parser = SVGParser(svg_file)
        elements = parser.parse()
        
        # 渲染位图
        start_time = time.time()
        bitmap = parser.render_to_bitmap(dpi=300)
        render_time = time.time() - start_time
        print(f"位图渲染耗时: {render_time:.2f}秒")
        
        if save_intermediate:
            bitmap_path = os.path.join(self.output_dir, "rendered_bitmap.png")
            cv2.imwrite(bitmap_path, cv2.cvtColor(bitmap, cv2.COLOR_RGB2BGR))
            print(f"保存渲染位图: {bitmap_path}")
        
        # 保存临时文件用于API
        temp_path = os.path.join(self.output_dir, "temp_for_api.png")
        cv2.imwrite(temp_path, cv2.cvtColor(bitmap, cv2.COLOR_RGB2BGR))
        
        # 调用API
        start_time = time.time()
        skin_mask = get_skin_mask(temp_path)
        api_time = time.time() - start_time
        print(f"API调用耗时: {api_time:.2f}秒")
        
        # 分析结果
        skin_pixels = np.sum(skin_mask > 0)
        total_pixels = skin_mask.shape[0] * skin_mask.shape[1]
        skin_ratio = skin_pixels / total_pixels
        
        print(f"\n皮肤分割结果:")
        print(f"  皮肤像素: {skin_pixels:,}")
        print(f"  总像素: {total_pixels:,}")
        print(f"  皮肤占比: {skin_ratio:.2%}")
        
        if save_intermediate:
            # 保存mask
            mask_path = os.path.join(self.output_dir, "skin_mask.png")
            cv2.imwrite(mask_path, skin_mask)
            print(f"保存皮肤mask: {mask_path}")
            
            # 创建可视化
            self._create_segmentation_visualization(bitmap, skin_mask)
        
        # 清理临时文件
        os.remove(temp_path)
        
        return skin_mask, skin_ratio
    
    def test_region_mapping(self, svg_file: str, skin_mask: np.ndarray):
        """测试区域映射功能"""
        print(f"\n测试区域映射...")
        
        parser = SVGParser(svg_file)
        elements = parser.parse()
        
        mapper = SVGRegionMapper(parser, skin_mask)
        skin_indices, env_indices = mapper.map_regions()
        
        print(f"映射结果:")
        print(f"  皮肤元素: {len(skin_indices)}")
        print(f"  环境元素: {len(env_indices)}")
        
        # 分析覆盖率分布
        coverage_stats = self._analyze_coverage_distribution(mapper.coverage_ratios)
        
        # 识别面部特征
        facial_features = []
        for idx in skin_indices:
            elem = elements[idx]
            if mapper._is_likely_facial_feature(elem):
                facial_features.append(idx)
        
        print(f"  可能的面部特征: {len(facial_features)}个")
        
        return skin_indices, env_indices, mapper
    
    def test_color_quantization(self, elements: List, 
                              skin_indices: List[int], 
                              env_indices: List[int],
                              k: int = 24):
        """测试颜色量化功能"""
        print(f"\n测试颜色量化 (目标: {k}色)...")
        
        quantizer = SVGColorQuantizer(use_gpu=True)
        
        # 准备数据
        all_colors = [elem.fill_color for elem in elements]
        skin_colors = [all_colors[i] for i in skin_indices]
        env_colors = [all_colors[i] for i in env_indices]
        
        # 计算颜色分配
        k_skin = max(1, int(k * len(skin_indices) / len(elements)))
        k_env = k - k_skin
        
        print(f"颜色分配: 皮肤{k_skin}色, 环境{k_env}色")
        
        # 量化
        start_time = time.time()
        
        if skin_colors:
            skin_labels, skin_centroids = quantizer.quantize_colors(skin_colors, k_skin)
            print(f"皮肤量化: {len(set(skin_labels))}种颜色")
        
        if env_colors:
            env_labels, env_centroids = quantizer.quantize_colors(env_colors, k_env)
            print(f"环境量化: {len(set(env_labels))}种颜色")
        
        quant_time = time.time() - start_time
        print(f"量化耗时: {quant_time:.2f}秒")
        
    def _plot_color_distribution(self, color_counts: Dict[Tuple[int, int, int], int]):
        """绘制颜色分布图"""
        colors = list(color_counts.keys())
        counts = list(color_counts.values())
        
        # 按数量排序，取前20
        sorted_data = sorted(zip(colors, counts), key=lambda x: x[1], reverse=True)[:20]
        colors, counts = zip(*sorted_data)
        
        # 创建颜色条
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # 柱状图
        x = range(len(colors))
        bars = ax1.bar(x, counts)
        
        # 设置颜色
        for i, (color, bar) in enumerate(zip(colors, bars)):
            bar.set_color(np.array(color) / 255.0)
        
        ax1.set_xlabel('颜色索引')
        ax1.set_ylabel('元素数量')
        ax1.set_title('SVG元素颜色分布（前20种）')
        
        # 颜色展示
        color_array = np.array(colors).reshape(1, -1, 3) / 255.0
        ax2.imshow(color_array, aspect='auto')
        ax2.set_yticks([])
        ax2.set_xticks(x)
        ax2.set_xlabel('颜色索引')
        ax2.set_title('颜色展示')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'color_distribution.png'))
        plt.close()
    
    def _create_segmentation_visualization(self, bitmap: np.ndarray, skin_mask: np.ndarray):
        """创建分割可视化"""
        # 创建叠加图
        overlay = bitmap.copy()
        skin_color = np.array([255, 0, 0])  # 红色标记皮肤
        overlay[skin_mask > 0] = overlay[skin_mask > 0] * 0.5 + skin_color * 0.5
        
        # 创建对比图
        h, w = bitmap.shape[:2]
        comparison = np.zeros((h, w * 3, 3), dtype=np.uint8)
        
        # 原图
        comparison[:, :w] = bitmap
        
        # Mask
        mask_colored = cv2.cvtColor(skin_mask, cv2.COLOR_GRAY2RGB)
        comparison[:, w:2*w] = mask_colored
        
        # 叠加
        comparison[:, 2*w:] = overlay.astype(np.uint8)
        
        # 添加标签
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(comparison, 'Original', (10, 30), font, 1, (255, 255, 255), 2)
        cv2.putText(comparison, 'Skin Mask', (w + 10, 30), font, 1, (255, 255, 255), 2)
        cv2.putText(comparison, 'Overlay', (2*w + 10, 30), font, 1, (255, 255, 255), 2)
        
        # 保存
        vis_path = os.path.join(self.output_dir, 'segmentation_visualization.png')
        cv2.imwrite(vis_path, cv2.cvtColor(comparison, cv2.COLOR_RGB2BGR))
        print(f"保存分割可视化: {vis_path}")
    
    def _analyze_coverage_distribution(self, coverage_ratios: Dict[int, float]) -> Dict[str, float]:
        """分析覆盖率分布"""
        coverages = list(coverage_ratios.values())
        
        stats = {
            'mean': np.mean(coverages),
            'std': np.std(coverages),
            'min': np.min(coverages),
            'max': np.max(coverages),
            'boundary_count': sum(1 for c in coverages if 0.3 <= c <= 0.7)
        }
        
        print(f"\n覆盖率统计:")
        print(f"  平均值: {stats['mean']:.3f}")
        print(f"  标准差: {stats['std']:.3f}")
        print(f"  范围: [{stats['min']:.3f}, {stats['max']:.3f}]")
        print(f"  边界元素: {stats['boundary_count']}个")
        
        # 绘制分布图
        plt.figure(figsize=(10, 6))
        plt.hist(coverages, bins=20, edgecolor='black')
        plt.axvline(0.5, color='r', linestyle='--', label='阈值(0.5)')
        plt.xlabel('覆盖率')
        plt.ylabel('元素数量')
        plt.title('元素皮肤覆盖率分布')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(self.output_dir, 'coverage_distribution.png'))
        plt.close()
        
        return stats

def run_complete_test(svg_file: str):
    """运行完整测试流程"""
    print("="*60)
    print(f"完整测试: {svg_file}")
    print("="*60)
    
    debugger = SVGDebugger()
    
    # 1. 分析SVG结构
    debugger.visualize_svg_elements(svg_file)
    
    # 2. 测试皮肤分割
    skin_mask, skin_ratio = debugger.test_skin_segmentation(svg_file)
    
    # 3. 测试区域映射
    parser = SVGParser(svg_file)
    elements = parser.parse()
    skin_indices, env_indices, mapper = debugger.test_region_mapping(svg_file, skin_mask)
    
    # 4. 测试颜色量化
    target_k = 24  # 或从文件名提取
    debugger.test_color_quantization(elements, skin_indices, env_indices, target_k)
    
    print("\n测试完成！")
    print(f"调试输出保存在: {debugger.output_dir}")

if __name__ == '__main__':
    # 测试示例
    test_svg = "input/test_image_24.svg"  # 替换为实际文件
    
    if os.path.exists(test_svg):
        run_complete_test(test_svg)
    else:
        print(f"测试文件不存在: {test_svg}")
        print("请将SVG文件放入input目录并更新路径")