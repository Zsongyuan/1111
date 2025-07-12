# svg_region_mapper.py
"""
SVG区域映射模块：将皮肤分割mask映射回SVG元素，并为所有元素计算重要性权重。
"""
import numpy as np
import cv2
from typing import List, Tuple, Dict
from svg_parser import SVGParser, SVGElement
import xml.etree.ElementTree as ET
try:
    import svg_config
except ImportError:
    class svg_config:
        PROCESSING_MODE = "fast"; REGION_MAPPING_CONFIG = {"fast": {"downsample_factor": 4, "batch_size": 100, "render_dpi": 75}, "accurate": {"downsample_factor": 2, "batch_size": 50, "render_dpi": 150}}

def brightness(color: Tuple[int, int, int]) -> float:
    r, g, b = color
    return 0.299 * float(r) + 0.587 * float(g) + 0.114 * float(b)

class SVGRegionMapper:
    """将皮肤mask映射到SVG元素"""
    
    def __init__(self, svg_parser: SVGParser, skin_mask: np.ndarray):
        self.svg_parser = svg_parser
        self.skin_mask = skin_mask
        self.coverage_ratios: Dict[int, float] = {}
        mode = getattr(svg_config, 'PROCESSING_MODE', 'fast')
        config = svg_config.REGION_MAPPING_CONFIG.get(mode, svg_config.REGION_MAPPING_CONFIG['fast'])
        self.downsample_factor = config['downsample_factor']
        self.batch_size = config['batch_size']
        self.skin_mask_small = self._downsample_mask(skin_mask)
        
    def _downsample_mask(self, mask: np.ndarray) -> np.ndarray:
        h, w = mask.shape
        new_h, new_w = h // self.downsample_factor, w // self.downsample_factor
        return cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        
    def map_regions(self, coverage_threshold: float = 0.5) -> Tuple[List[int], List[int], List[int], List[int]]:
        skin_indices, initial_env_indices = [], []
        print("开始批量计算元素覆盖率...")
        self._batch_calculate_coverage()
        
        for element in self.svg_parser.elements:
            coverage = self.coverage_ratios.get(element.index, 0)
            if coverage >= coverage_threshold:
                skin_indices.append(element.index)
            else:
                initial_env_indices.append(element.index)

        print("从非皮肤区域中识别眼睛和嘴部特征...")
        env_indices, eye_indices, mouth_indices = self.separate_facial_features(initial_env_indices)
        
        print(f"区域分离完成：皮肤({len(skin_indices)}), 环境({len(env_indices)}), 眼睛({len(eye_indices)}), 嘴巴({len(mouth_indices)})")
        return skin_indices, env_indices, eye_indices, mouth_indices

    def separate_facial_features(self, non_skin_indices: List[int]) -> Tuple[List[int], List[int], List[int]]:
        eye_indices, mouth_indices = [], []
        contours, _ = cv2.findContours(self.skin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return non_skin_indices, [], []
            
        main_face_contour = max(contours, key=cv2.contourArea)
        face_x, face_y, face_w, face_h = cv2.boundingRect(main_face_contour)
        non_skin_elements = [e for e in self.svg_parser.elements if e.index in non_skin_indices and e.bbox]
        remaining_env_indices = set(non_skin_indices)
        
        for elem in non_skin_elements:
            ex, ey, _, _ = elem.bbox
            if not (face_x < ex < face_x + face_w and face_y < ey < face_y + face_h):
                continue
            r, g, b = elem.fill_color
            elem_brightness = brightness(elem.fill_color)
            is_white_like = r > 220 and g > 220 and b > 220
            
            if is_white_like and ey > face_y + face_h * 0.6:
                mouth_indices.append(elem.index)
                if elem.index in remaining_env_indices: remaining_env_indices.remove(elem.index)
                continue
            
            if ey < face_y + face_h * 0.6:
                is_dark = elem_brightness < 40
                hsv = cv2.cvtColor(np.uint8([[list(elem.fill_color)]]), cv2.COLOR_RGB2HSV)[0][0]
                is_colorful_iris = hsv[1] > 60 and not is_white_like
                if is_white_like or is_dark or is_colorful_iris:
                    eye_indices.append(elem.index)
                    if elem.index in remaining_env_indices: remaining_env_indices.remove(elem.index)
        return list(remaining_env_indices), eye_indices, mouth_indices

    def get_element_importance_weights(self, skin_indices, eye_indices, mouth_indices) -> Dict[int, float]:
        """【已重构】为所有元素生成统一的重要性权重字典。"""
        print("生成统一重要性权重...")
        weights = {}
        skin_elements = [e for e in self.svg_parser.elements if e.index in skin_indices]
        
        # 默认权重
        for element in self.svg_parser.elements:
            weights[element.index] = 1.0

        # 边界权重
        for idx, coverage in self.coverage_ratios.items():
            if 0.3 <= coverage <= 0.7:
                weights[idx] = 1.5

        # 重要皮肤特征权重
        if skin_elements:
            all_brightness = [brightness(e.fill_color) for e in skin_elements]
            brightness_threshold = np.percentile(all_brightness, 10)
            for elem in skin_elements:
                if brightness(elem.fill_color) < brightness_threshold:
                    weights[elem.index] = 5.0
        
        # 最高权重：眼睛和嘴巴
        for idx in eye_indices:
            weights[idx] = 20.0
        for idx in mouth_indices:
            weights[idx] = 20.0
            
        return weights
    
    def _batch_calculate_coverage(self):
        # ... 此方法无变化 ...
        elements_list = list(self.svg_parser.elements)
        for i in range(0, len(elements_list), self.batch_size):
            batch = elements_list[i:i + self.batch_size]
            self._process_batch_color_method(batch)
            print(f"  进度: {min(100, (i + self.batch_size) * 100 // len(elements_list))}%", end='\r')
        print("\n覆盖率计算完成")

    def _process_batch_color_method(self, batch: List[SVGElement]):
        # ... 此方法无变化 ...
        import cairosvg
        from io import BytesIO
        from PIL import Image
        h, w = self.skin_mask_small.shape
        svg = ET.Element('svg', {'xmlns': 'http://www.w3.org/2000/svg', 'width': str(self.svg_parser.width), 'height': str(self.svg_parser.height), 'viewBox': f'0 0 {self.svg_parser.width} {self.svg_parser.height}'})
        ET.SubElement(svg, 'rect', {'x': '0', 'y': '0', 'width': str(self.svg_parser.width), 'height': str(self.svg_parser.height), 'fill': 'white'})
        color_to_idx = {}
        for i, elem in enumerate(batch):
            color_idx = i + 1; r,g,b = (color_idx>>16)&0xFF, (color_idx>>8)&0xFF, color_idx&0xFF
            path = ET.SubElement(svg, 'path', {'d': elem.path_data, 'fill': f'#{r:02x}{g:02x}{b:02x}'})
            if elem.transform: path.set('transform', elem.transform)
            color_to_idx[color_idx] = elem.index
        try:
            h_orig, w_orig = self.skin_mask.shape
            png_data = cairosvg.svg2png(bytestring=ET.tostring(svg), output_width=w_orig, output_height=h_orig)
            rendered_rgb = cv2.resize(np.array(Image.open(BytesIO(png_data)).convert('RGB')), (w, h), interpolation=cv2.INTER_NEAREST)
            for color_idx, elem_idx in color_to_idx.items():
                r,g,b = (color_idx>>16)&0xFF, (color_idx>>8)&0xFF, color_idx&0xFF
                color_mask = np.all(rendered_rgb == [r, g, b], axis=-1)
                element_area = np.sum(color_mask)
                if element_area > 0:
                    self.coverage_ratios[elem_idx] = np.sum(color_mask & (self.skin_mask_small > 0)) / element_area
                else: self.coverage_ratios[elem_idx] = 0.0
        except Exception as e:
            print(f"批量渲染失败: {e}"); [self.coverage_ratios.setdefault(elem.index, 0.5) for elem in batch]

class FastSVGRegionMapper(SVGRegionMapper):
    # ... 此类无变化 ...
    def __init__(self, svg_parser: SVGParser, skin_mask: np.ndarray):
        super().__init__(svg_parser, skin_mask)
    def _batch_calculate_coverage(self):
        print("使用快速边界框采样方法计算覆盖率...")
        h_small, w_small = self.skin_mask_small.shape
        svg_h, svg_w = self.svg_parser.height, self.svg_parser.width
        if svg_h == 0 or svg_w == 0: return [self.coverage_ratios.setdefault(e.index, 0.0) for e in self.svg_parser.elements]
        x_scale, y_scale = w_small / svg_w, h_small / svg_h
        for element in self.svg_parser.elements:
            if not element.bbox or element.bbox[2] <= 0 or element.bbox[3] <= 0: self.coverage_ratios[element.index] = 0.0; continue
            xmin_svg, ymin_svg, w_svg, h_svg = element.bbox
            x_start, y_start = int(xmin_svg*x_scale), int(ymin_svg*y_scale)
            x_end, y_end = int((xmin_svg+w_svg)*x_scale), int((ymin_svg+h_svg)*y_scale)
            x_start, y_start, x_end, y_end = max(0,x_start), max(0,y_start), min(w_small,x_end), min(h_small,y_end)
            roi = self.skin_mask_small[y_start:y_end, x_start:x_end]
            self.coverage_ratios[element.index] = np.sum(roi > 0) / roi.size if roi.size > 0 else 0.0
        print("快速覆盖率计算完成")