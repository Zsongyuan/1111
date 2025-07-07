# svg_parser.py
"""
SVG解析模块：解析SVG文件，提取path元素的颜色、透明度和路径数据
"""
import xml.etree.ElementTree as ET
import numpy as np
import re
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
import cv2
from svgpathtools import parse_path, Path
import cairosvg
from io import BytesIO
from PIL import Image

@dataclass
class SVGElement:
    """SVG路径元素数据结构"""
    index: int
    path_data: str
    fill_color: Tuple[int, int, int]  # RGB
    opacity: float
    stroke_color: Optional[Tuple[int, int, int]] = None
    stroke_width: float = 0
    transform: Optional[str] = None
    bbox: Optional[Tuple[float, float, float, float]] = None  # x, y, width, height

class SVGParser:
    """SVG文件解析器"""
    
    def __init__(self, svg_file_path: str):
        self.svg_file_path = svg_file_path
        self.elements: List[SVGElement] = []
        self.width = 0
        self.height = 0
        self.tree = None
        self.root = None
        
    def parse(self) -> List[SVGElement]:
        """解析SVG文件，返回所有path元素列表"""
        self.tree = ET.parse(self.svg_file_path)
        self.root = self.tree.getroot()
        
        # 获取SVG尺寸
        self._extract_dimensions()
        
        # 提取所有path元素
        self._extract_path_elements()
        
        return self.elements
    
    def _extract_dimensions(self):
        """提取SVG画布尺寸"""
        viewbox = self.root.get('viewBox')
        if viewbox:
            parts = viewbox.split()
            if len(parts) == 4:
                self.width = float(parts[2])
                self.height = float(parts[3])
        else:
            width = self.root.get('width', '0')
            height = self.root.get('height', '0')
            self.width = float(re.findall(r'[\d.]+', width)[0]) if re.findall(r'[\d.]+', width) else 0
            self.height = float(re.findall(r'[\d.]+', height)[0]) if re.findall(r'[\d.]+', height) else 0
    
    def _extract_path_elements(self):
        """递归提取所有path元素"""
        namespaces = {'svg': 'http://www.w3.org/2000/svg'}
        
        # 查找所有path元素
        for idx, path_elem in enumerate(self.root.findall('.//svg:path', namespaces) or self.root.findall('.//path')):
            element = self._parse_path_element(path_elem, idx)
            if element:
                self.elements.append(element)
    
    def _parse_path_element(self, path_elem, index: int) -> Optional[SVGElement]:
        """解析单个path元素"""
        path_data = path_elem.get('d', '')
        if not path_data:
            return None
        
        # 解析填充颜色
        fill = path_elem.get('fill', '#000000')
        fill_color = self._parse_color(fill)
        if fill_color is None:  # 如果是none或transparent，跳过
            return None
        
        # 解析透明度
        opacity = float(path_elem.get('opacity', '1.0'))
        fill_opacity = float(path_elem.get('fill-opacity', '1.0'))
        total_opacity = opacity * fill_opacity
        
        # 解析描边
        stroke = path_elem.get('stroke', 'none')
        stroke_color = self._parse_color(stroke) if stroke != 'none' else None
        stroke_width = float(path_elem.get('stroke-width', '0'))
        
        # 获取变换
        transform = path_elem.get('transform')
        
        # 计算边界框
        try:
            path = parse_path(path_data)
            bbox = self._calculate_bbox(path)
        except:
            bbox = None
        
        return SVGElement(
            index=index,
            path_data=path_data,
            fill_color=fill_color,
            opacity=total_opacity,
            stroke_color=stroke_color,
            stroke_width=stroke_width,
            transform=transform,
            bbox=bbox
        )
    
    def _parse_color(self, color_str: str) -> Optional[Tuple[int, int, int]]:
        """解析颜色字符串，返回RGB元组"""
        if not color_str or color_str.lower() in ['none', 'transparent']:
            return None
        
        # 处理十六进制颜色
        if color_str.startswith('#'):
            if len(color_str) == 4:  # #RGB
                r = int(color_str[1] * 2, 16)
                g = int(color_str[2] * 2, 16)
                b = int(color_str[3] * 2, 16)
            elif len(color_str) == 7:  # #RRGGBB
                r = int(color_str[1:3], 16)
                g = int(color_str[3:5], 16)
                b = int(color_str[5:7], 16)
            else:
                return (0, 0, 0)
            return (r, g, b)
        
        # 处理rgb()格式
        rgb_match = re.match(r'rgb\s*\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)', color_str)
        if rgb_match:
            return tuple(int(x) for x in rgb_match.groups())
        
        # 处理命名颜色（简化版本，只处理常见颜色）
        named_colors = {
            'black': (0, 0, 0),
            'white': (255, 255, 255),
            'red': (255, 0, 0),
            'green': (0, 128, 0),
            'blue': (0, 0, 255),
            'yellow': (255, 255, 0),
            'cyan': (0, 255, 255),
            'magenta': (255, 0, 255),
        }
        return named_colors.get(color_str.lower(), (0, 0, 0))
    
    def _calculate_bbox(self, path: Path) -> Tuple[float, float, float, float]:
        """计算路径的边界框"""
        try:
            xmin, xmax, ymin, ymax = path.bbox()
            return (xmin, ymin, xmax - xmin, ymax - ymin)
        except:
            return (0, 0, 0, 0)
    
    def render_to_bitmap(self, dpi: int = 300) -> np.ndarray:
        """将SVG渲染为位图（用于皮肤分割）"""
        # 使用svgConvertor模块，它包含尺寸检查和去噪功能
        from svgConvertor import convert_and_denoise
        processed_image, _ = convert_and_denoise(self.svg_file_path, dpi)
        # convert_and_denoise返回BGR格式，转换为RGB
        return cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB)
    
    def create_element_mask(self, element_indices: List[int], image_shape: Tuple[int, int]) -> np.ndarray:
        """为指定的元素创建mask"""
        mask = np.zeros(image_shape[:2], dtype=np.uint8)
        
        # 创建临时SVG只包含指定元素
        temp_root = ET.Element(self.root.tag, self.root.attrib)
        
        for elem in self.elements:
            if elem.index in element_indices:
                # 创建path元素
                path_elem = ET.SubElement(temp_root, 'path')
                path_elem.set('d', elem.path_data)
                path_elem.set('fill', 'white')
                if elem.transform:
                    path_elem.set('transform', elem.transform)
        
        # 渲染临时SVG
        svg_str = ET.tostring(temp_root, encoding='unicode')
        png_data = cairosvg.svg2png(bytestring=svg_str.encode('utf-8'))
        image = Image.open(BytesIO(png_data))
        mask_image = np.array(image.convert('L'))
        
        # 调整大小以匹配目标尺寸
        if mask_image.shape != image_shape[:2]:
            mask_image = cv2.resize(mask_image, (image_shape[1], image_shape[0]), interpolation=cv2.INTER_NEAREST)
        
        return (mask_image > 128).astype(np.uint8) * 255
    
    def update_element_colors(self, color_mapping: Dict[int, Tuple[int, int, int]]):
        """更新元素颜色"""
        for elem_idx, new_color in color_mapping.items():
            if elem_idx < len(self.elements):
                self.elements[elem_idx].fill_color = new_color
    
    def save_svg(self, output_path: str):
        """保存更新后的SVG文件"""
        # 创建索引映射
        elem_map = {elem.index: elem for elem in self.elements}
        
        # 更新XML树中的颜色
        namespaces = {'svg': 'http://www.w3.org/2000/svg'}
        path_elements = list(self.root.findall('.//svg:path', namespaces) or self.root.findall('.//path'))
        
        for idx, path_elem in enumerate(path_elements):
            if idx in elem_map:
                new_color = elem_map[idx].fill_color
                color_str = f'#{new_color[0]:02x}{new_color[1]:02x}{new_color[2]:02x}'
                path_elem.set('fill', color_str)
        
        # 保存文件
        self.tree.write(output_path, encoding='utf-8', xml_declaration=True)