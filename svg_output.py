# svg_output.py
"""
SVG输出模块：生成更新颜色后的SVG文件和位图
"""
import xml.etree.ElementTree as ET
import numpy as np
import cv2
from typing import Dict, Tuple
from svg_parser import SVGParser
import os

class SVGOutput:
    """SVG输出生成器"""
    
    def __init__(self, svg_parser: SVGParser):
        self.svg_parser = svg_parser
        self.updated_tree = None
        
    def update_svg_colors(self, color_mapping: Dict[int, Tuple[int, int, int]]):
        """更新SVG中的颜色"""
        # 深拷贝原始树以进行修改
        self.updated_tree = ET.parse(self.svg_parser.svg_file_path)
        root = self.updated_tree.getroot()
        
        namespaces = {'svg': 'http://www.w3.org/2000/svg'}
        
        # 优先使用带命名空间的查找，如果失败则回退到不带命名空间的查找
        path_elements = root.findall('.//svg:path', namespaces)
        if not path_elements:
            path_elements = root.findall('.//path')
            
        for idx, new_color in color_mapping.items():
            if idx < len(path_elements):
                path_elem = path_elements[idx]
                if isinstance(new_color, (list, tuple)) and len(new_color) >= 3:
                    color_str = self._rgb_to_hex(new_color)
                    path_elem.set('fill', color_str)
                else:
                    # 对于无效的颜色，可以选择跳过或设置为默认颜色，这里我们选择跳过
                    print(f"警告: 元素索引 {idx} 的颜色格式不正确: {new_color}，已跳过。")
        
    def save_svg(self, output_path: str):
        """保存更新后的SVG文件"""
        if self.updated_tree is None:
            raise ValueError("必须先调用update_svg_colors()更新颜色")
        
        # 【核心修复】在这里注册默认命名空间，以移除保存时的'svg:'前缀
        ET.register_namespace('', "http://www.w3.org/2000/svg")
        
        # 为了美观，可以对XML进行缩进处理
        self._indent_tree(self.updated_tree.getroot())
        
        # 写入文件
        self.updated_tree.write(output_path, 
                               encoding='utf-8', 
                               xml_declaration=True,
                               method='xml')
        
        print(f"SVG文件已保存: {output_path}")
    
    def _rgb_to_hex(self, rgb: Tuple[int, int, int]) -> str:
        """将RGB元组（可能包含浮点数）转换为十六进制颜色字符串"""
        # 处理可能的浮点数和边界值
        r = max(0, min(255, int(round(float(rgb[0])))))
        g = max(0, min(255, int(round(float(rgb[1])))))
        b = max(0, min(255, int(round(float(rgb[2])))))
        
        # 确保格式正确（两位十六进制）
        return f'#{r:02x}{g:02x}{b:02x}'
    
    def _indent_tree(self, elem, level=0):
        """为XML树添加漂亮的缩进，使其更具可读性"""
        i = "\n" + level*"  "
        if len(elem):
            if not elem.text or not elem.text.strip():
                elem.text = i + "  "
            if not elem.tail or not elem.tail.strip():
                elem.tail = i
            for subelem in elem:
                self._indent_tree(subelem, level+1)
            if not subelem.tail or not subelem.tail.strip():
                subelem.tail = i
        else:
            if level and (not elem.tail or not elem.tail.strip()):
                elem.tail = i

class BatchSVGProcessor:
    """批量SVG处理器"""
    
    def __init__(self):
        pass

    def create_comparison_image(self, original_path: str, processed_path: str, output_path: str, dpi: int = 300):
        """创建原始与处理后的对比图"""
        from svgConvertor import convert_and_denoise
        
        # 渲染原始图像
        original_img, _ = convert_and_denoise(original_path, dpi)
        original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
        
        # 渲染处理后图像
        processed_img, _ = convert_and_denoise(processed_path, dpi)
        processed_img = cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB)
        
        # 统一尺寸
        h1, w1 = original_img.shape[:2]
        h2, w2 = processed_img.shape[:2]
        h, w = max(h1, h2), max(w1, w2)
        
        if (h1, w1) != (h, w):
            original_img = cv2.resize(original_img, (w, h))
        if (h2, w2) != (h, w):
            processed_img = cv2.resize(processed_img, (w, h))
        
        # 创建并排对比
        comparison = np.hstack([original_img, processed_img])
        
        # 添加标签
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(comparison, 'Original', (10, 30), font, 1, (255, 255, 255), 2)
        cv2.putText(comparison, 'Processed', (w + 10, 30), font, 1, (255, 255, 255), 2)
        
        # 保存对比图
        cv2.imwrite(output_path, cv2.cvtColor(comparison, cv2.COLOR_RGB2BGR))
        print(f"对比图像已保存: {output_path}")

    def generate_color_palette_image(self, color_mapping: Dict[int, Tuple[int, int, int]], output_path: str, block_size: int = 50):
        """生成色板图像"""
        unique_colors = list(set(color_mapping.values()))
        n_colors = len(unique_colors)
        
        if n_colors == 0:
            return
        
        # 计算网格尺寸
        cols = min(10, n_colors)
        rows = (n_colors + cols - 1) // cols
        
        # 创建色板图像
        palette_img = np.zeros((rows * block_size, cols * block_size, 3), dtype=np.uint8)
        
        for i, color in enumerate(unique_colors):
            row, col = i // cols, i % cols
            y_start, y_end = row * block_size, (row + 1) * block_size
            x_start, x_end = col * block_size, (col + 1) * block_size
            palette_img[y_start:y_end, x_start:x_end] = color
        
        # 保存色板图像
        cv2.imwrite(output_path, cv2.cvtColor(palette_img, cv2.COLOR_RGB2BGR))
        print(f"色板图像已保存: {output_path} (包含 {n_colors} 种颜色)")