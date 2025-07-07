# svg_output.py
"""
SVG输出模块：生成更新颜色后的SVG文件和位图
"""
import xml.etree.ElementTree as ET
import numpy as np
import cv2
import cairosvg
from typing import Dict, Tuple, Optional
from svg_parser import SVGParser
from io import BytesIO
from PIL import Image
import os

class SVGOutput:
    """SVG输出生成器"""
    
    def __init__(self, svg_parser: SVGParser):
        self.svg_parser = svg_parser
        self.updated_tree = None
        
    def update_svg_colors(self, color_mapping: Dict[int, Tuple[int, int, int]]) -> ET.ElementTree:
        """
        更新SVG中的颜色
        
        参数:
            color_mapping: 元素索引到新颜色的映射
            
        返回:
            更新后的XML树
        """
        # 深拷贝原始树
        self.updated_tree = ET.parse(self.svg_parser.svg_file_path)
        root = self.updated_tree.getroot()
        
        # 获取所有path元素
        namespaces = {'svg': 'http://www.w3.org/2000/svg'}
        path_elements = list(root.findall('.//svg:path', namespaces))
        if not path_elements:
            path_elements = list(root.findall('.//path'))
        
        # 更新颜色
        for idx, new_color in color_mapping.items():
            if idx < len(path_elements):
                path_elem = path_elements[idx]
                # 确保颜色是正确的格式
                if isinstance(new_color, (list, tuple)) and len(new_color) >= 3:
                    color_str = self._rgb_to_hex(new_color)
                    path_elem.set('fill', color_str)
                else:
                    print(f"警告: 元素 {idx} 的颜色格式不正确: {new_color}")
        
        return self.updated_tree
    
    def save_svg(self, output_path: str):
        """保存更新后的SVG文件"""
        if self.updated_tree is None:
            raise ValueError("必须先调用update_svg_colors()更新颜色")
        
        # 格式化输出
        self._indent_tree(self.updated_tree.getroot())
        
        # 写入文件
        self.updated_tree.write(output_path, 
                               encoding='utf-8', 
                               xml_declaration=True,
                               method='xml')
        
        print(f"SVG文件已保存: {output_path}")
    
    def render_to_bitmap(self, output_path: str, dpi: int = 300) -> np.ndarray:
        """将更新后的SVG渲染为位图"""
        if self.updated_tree is None:
            raise ValueError("必须先调用update_svg_colors()更新颜色")
        
        # 先保存临时SVG文件
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.svg', delete=False) as tmp:
            self._indent_tree(self.updated_tree.getroot())
            self.updated_tree.write(tmp.name, encoding='utf-8', xml_declaration=True, method='xml')
            temp_svg_path = tmp.name
        
        try:
            # 使用svgConvertor渲染
            from svgConvertor import convert_and_denoise
            bitmap_bgr, _ = convert_and_denoise(temp_svg_path, dpi)
            
            # 保存位图
            cv2.imwrite(output_path, bitmap_bgr)
            print(f"位图已保存: {output_path}")
            
            # 转换为RGB格式返回
            bitmap = cv2.cvtColor(bitmap_bgr, cv2.COLOR_BGR2RGB)
            return bitmap
            
        finally:
            # 清理临时文件
            import os
            os.unlink(temp_svg_path)
    
    def _rgb_to_hex(self, rgb: Tuple[int, int, int]) -> str:
        """将RGB元组转换为十六进制颜色字符串"""
        # 确保RGB值是整数
        r = int(rgb[0])
        g = int(rgb[1])
        b = int(rgb[2])
        return f'#{r:02x}{g:02x}{b:02x}'
    
    def _indent_tree(self, elem, level=0):
        """格式化XML树的缩进"""
        indent = "\n" + "  " * level
        if len(elem):
            if not elem.text or not elem.text.strip():
                elem.text = indent + "  "
            if not elem.tail or not elem.tail.strip():
                elem.tail = indent
            for child in elem:
                self._indent_tree(child, level + 1)
            if not child.tail or not child.tail.strip():
                child.tail = indent
        else:
            if level and (not elem.tail or not elem.tail.strip()):
                elem.tail = indent

class BatchSVGProcessor:
    """批量SVG处理器"""
    
    def __init__(self):
        pass
    
    # 文件: svg_output.py

    def create_comparison_image(self, 
                               original_path: str,
                               processed_path: str,
                               output_path: str,
                               dpi: int = 300):
        """创建对比图像（原图和处理后图像并排）"""
        from svgConvertor import convert_and_denoise
        
        # 读取原图
        original_img, _ = convert_and_denoise(original_path, dpi)
        original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
        
        # 读取处理后图像  
        processed_img, _ = convert_and_denoise(processed_path, dpi)
        processed_img = cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB)
        
        # 确保尺寸一致
        h1, w1 = original_img.shape[:2]
        h2, w2 = processed_img.shape[:2]
        
        # --- 修复开始 ---
        # 预先定义 h 和 w，确保它们总是有值
        h, w = h1, w1
        # --- 修复结束 ---
        
        if (h1, w1) != (h2, w2):
            # 调整到相同尺寸
            h = max(h1, h2)
            w = max(w1, w2)
            
            if (h1, w1) != (h, w):
                original_img = cv2.resize(original_img, (w, h))
            if (h2, w2) != (h, w):
                processed_img = cv2.resize(processed_img, (w, h))
        
        # 创建对比图
        comparison = np.hstack([original_img, processed_img])
        
        # 添加标签
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(comparison, 'Original', (10, 30), font, 1, (255, 255, 255), 2)
        cv2.putText(comparison, 'Processed', (w + 10, 30), font, 1, (255, 255, 255), 2)
        
        # 保存
        comparison_bgr = cv2.cvtColor(comparison, cv2.COLOR_RGB2BGR)
        cv2.imwrite(output_path, comparison_bgr)
        
        print(f"对比图像已保存: {output_path}")
    
    def generate_color_palette_image(self,
                                   color_mapping: Dict[int, Tuple[int, int, int]],
                                   output_path: str,
                                   block_size: int = 50):
        """生成颜色色板图像"""
        unique_colors = list(set(color_mapping.values()))
        n_colors = len(unique_colors)
        
        if n_colors == 0:
            return
        
        # 计算图像尺寸
        cols = min(10, n_colors)
        rows = (n_colors + cols - 1) // cols
        
        # 创建图像
        img_height = rows * block_size
        img_width = cols * block_size
        palette_img = np.zeros((img_height, img_width, 3), dtype=np.uint8)
        
        # 填充颜色块
        for i, color in enumerate(unique_colors):
            row = i // cols
            col = i % cols
            
            y1 = row * block_size
            y2 = (row + 1) * block_size
            x1 = col * block_size
            x2 = (col + 1) * block_size
            
            palette_img[y1:y2, x1:x2] = color
        
        # 保存
        palette_bgr = cv2.cvtColor(palette_img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(output_path, palette_bgr)
        
        print(f"色板图像已保存: {output_path} (包含 {n_colors} 种颜色)")