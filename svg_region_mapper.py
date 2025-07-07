# svg_region_mapper.py
"""
SVG区域映射模块：将皮肤分割mask映射回SVG元素，确定每个元素属于皮肤区域还是环境区域
优化版本：使用批量渲染和降采样加速处理
"""
import numpy as np
import cv2
from typing import List, Tuple, Dict, Set
from svg_parser import SVGParser, SVGElement
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
import threading
try:
    import svg_config
except ImportError:
    # 默认配置
    class svg_config:
        PROCESSING_MODE = "fast"
        REGION_MAPPING_CONFIG = {
            "fast": {"downsample_factor": 4, "batch_size": 100, "render_dpi": 75},
            "accurate": {"downsample_factor": 2, "batch_size": 50, "render_dpi": 150}
        }

def brightness(color: Tuple[int, int, int]) -> float:
    """计算颜色亮度"""
    r, g, b = color
    return 0.299 * float(r) + 0.587 * float(g) + 0.114 * float(b)

class SVGRegionMapper:
    """将皮肤mask映射到SVG元素"""
    
    def __init__(self, svg_parser: SVGParser, skin_mask: np.ndarray):
        self.svg_parser = svg_parser
        self.skin_mask = skin_mask
        self.element_masks: Dict[int, np.ndarray] = {}
        self.coverage_ratios: Dict[int, float] = {}
        # 从配置获取参数
        mode = getattr(svg_config, 'PROCESSING_MODE', 'fast')
        config = svg_config.REGION_MAPPING_CONFIG.get(mode, svg_config.REGION_MAPPING_CONFIG['fast'])
        self.downsample_factor = config['downsample_factor']
        self.batch_size = config['batch_size']
        self.render_dpi = config['render_dpi']
        # 降采样后的mask
        self.skin_mask_small = self._downsample_mask(skin_mask)
        
    def _downsample_mask(self, mask: np.ndarray) -> np.ndarray:
        """降采样mask以加速计算"""
        h, w = mask.shape
        new_h = h // self.downsample_factor
        new_w = w // self.downsample_factor
        return cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        
    def map_regions(self, coverage_threshold: float = 0.5) -> Tuple[List[int], List[int]]:
        """
        将SVG元素映射到皮肤/环境区域 (已修复参数传递问题)
        """
        skin_indices = []
        env_indices = []
        
        print("开始批量计算元素覆盖率...")
        self._batch_calculate_coverage()
        
        for element in self.svg_parser.elements:
            coverage = self.coverage_ratios.get(element.index, 0)
            if coverage >= coverage_threshold:
                skin_indices.append(element.index)
            else:
                env_indices.append(element.index)
        
        # --- 修复核心 ---
        # 1. 在调用细化函数前，先收集好所有皮肤元素
        all_skin_svg_elements = [e for e in self.svg_parser.elements if e.index in skin_indices]
        
        # 2. 将皮肤元素列表作为参数传递给细化函数
        skin_indices, env_indices = self._refine_boundary_elements(skin_indices, env_indices, all_skin_svg_elements)
        # --- 修复结束 ---
        
        return skin_indices, env_indices

    def _refine_boundary_elements(self, skin_indices: List[int], env_indices: List[int], all_skin_elements: List[SVGElement]) -> Tuple[List[int], List[int]]:
        """
        细化边界元素的分类 (已修复参数传递问题)
        """
        boundary_elements = []
        for idx, coverage in self.coverage_ratios.items():
            if 0.3 <= coverage <= 0.7:
                boundary_elements.append(idx)
        
        for elem_idx in boundary_elements:
            element = next((e for e in self.svg_parser.elements if e.index == elem_idx), None)
            if not element:
                continue

            # --- 修复核心 ---
            # 在调用时，传入第二个必需的参数
            if self._is_likely_facial_feature(element, all_skin_elements):
            # --- 修复结束 ---
                if elem_idx in env_indices:
                    env_indices.remove(elem_idx)
                    skin_indices.append(elem_idx)
        
        return skin_indices, env_indices
    
    def _batch_calculate_coverage(self):
        """批量计算所有元素的覆盖率"""
        # 方案1：使用颜色标记法批量渲染
        elements_list = list(self.svg_parser.elements)
        
        for i in range(0, len(elements_list), self.batch_size):
            batch = elements_list[i:i + self.batch_size]
            self._process_batch_color_method(batch, i)
            
            # 显示进度
            progress = min(100, (i + self.batch_size) * 100 // len(elements_list))
            print(f"  进度: {progress}%", end='\r')
        
        print("\n覆盖率计算完成")
    
    def _process_batch_color_method(self, batch: List[SVGElement], start_idx: int):
        """
        使用颜色标记法批量处理元素 (已优化对齐问题)
        """
        # 导入所需模块
        import cairosvg
        from io import BytesIO
        from PIL import Image

        # h, w 是降采样后的小尺寸
        h, w = self.skin_mask_small.shape
        
        # 创建一个SVG，每个元素使用唯一颜色
        # (这部分逻辑保持不变)
        svg = ET.Element('svg')
        svg.set('xmlns', 'http://www.w3.org/2000/svg')
        svg.set('width', str(self.svg_parser.width))
        svg.set('height', str(self.svg_parser.height))
        svg.set('viewBox', f'0 0 {self.svg_parser.width} {self.svg_parser.height}')
        
        bg = ET.SubElement(svg, 'rect')
        bg.set('x', '0')
        bg.set('y', '0')
        bg.set('width', str(self.svg_parser.width))
        bg.set('height', str(self.svg_parser.height))
        bg.set('fill', 'white')
        
        color_to_idx = {}
        for i, elem in enumerate(batch):
            color_idx = i + 1
            r = (color_idx >> 16) & 0xFF
            g = (color_idx >> 8) & 0xFF
            b = color_idx & 0xFF
            color_str = f'#{r:02x}{g:02x}{b:02x}'
            
            path = ET.SubElement(svg, 'path')
            path.set('d', elem.path_data)
            path.set('fill', color_str)
            if elem.transform:
                path.set('transform', elem.transform)
            
            color_to_idx[color_idx] = elem.index
        
        # --- 核心修复 ---
        # 渲染SVG到内存，并强制其输出尺寸与原始皮肤蒙版完全一致，以确保像素级对齐
        svg_str = ET.tostring(svg, encoding='unicode')
        
        try:
            # 获取原始（未降采样）皮肤蒙版的尺寸
            h_orig, w_orig = self.skin_mask.shape
            
            # 使用cairosvg直接渲染到指定尺寸的PNG数据
            png_data = cairosvg.svg2png(bytestring=svg_str.encode('utf-8'), output_width=w_orig, output_height=h_orig)
            
            # 从内存中的PNG数据加载图像
            image = Image.open(BytesIO(png_data))
            rendered_rgb_full = np.array(image.convert('RGB'))

            # 现在，我们有了一个与原始皮肤蒙版完美对齐的元素映射图。
            # 接着，使用与处理皮肤蒙版完全相同的方法，将其降采样到小尺寸。
            rendered_rgb = cv2.resize(rendered_rgb_full, (w, h), interpolation=cv2.INTER_NEAREST)
            # --- 修复结束 ---

            # 分析每个颜色的覆盖率 (这部分逻辑保持不变)
            for color_idx, elem_idx in color_to_idx.items():
                r = (color_idx >> 16) & 0xFF
                g = (color_idx >> 8) & 0xFF
                b = color_idx & 0xFF
                
                color_mask = np.all(rendered_rgb == [r, g, b], axis=-1)
                
                element_area = np.sum(color_mask)
                if element_area > 0:
                    intersection = np.sum(color_mask & (self.skin_mask_small > 0))
                    coverage = intersection / element_area
                else:
                    coverage = 0.0
                
                self.coverage_ratios[elem_idx] = coverage
                
        except Exception as e:
            print(f"批量渲染失败: {e}")
            for elem in batch:
                self.coverage_ratios[elem.index] = 0.5  # 降级使用默认值
    
    
    def _is_likely_facial_feature(self, element: SVGElement, all_skin_elements: List[SVGElement]) -> bool:
        """
        通过统计分析和颜色模型，更精确地判断元素是否为关键面部特征（眼睛、嘴唇）。
        这是一个更鲁棒的实现，以避免过度保留颜色。
        """
        if not all_skin_elements:
            return False

        # --- 1. 识别眼睛：通常是皮肤区域内最暗的颜色 ---
        # 计算所有皮肤元素的亮度
        all_brightness = [brightness(e.fill_color) for e in all_skin_elements]
        # 设置一个阈值，例如，只有亮度排在最暗的前5%的元素才可能是眼睛
        brightness_threshold = np.percentile(all_brightness, 5) 
        
        current_brightness = brightness(element.fill_color)
        
        # 如果当前元素的亮度低于阈值，且颜色偏向黑/灰/深棕，则有可能是眼睛
        if current_brightness < brightness_threshold and current_brightness < 60:
             return True

        # --- 2. 识别嘴唇：通常是高饱和度的红色/粉色 ---
        r, g, b = element.fill_color
        # 使用简化的HSV计算饱和度
        max_val = max(r, g, b)
        min_val = min(r, g, b)
        saturation = (max_val - min_val) / max_val if max_val > 0 else 0
        
        # 必须是红色调 (R分量最高) 且饱和度足够高
        is_reddish = r > g and r > b
        is_saturated = saturation > 0.4 

        if is_reddish and is_saturated:
            # 进一步检查，只有饱和度排在前10%的红色元素才被认为是嘴唇
            all_red_saturations = [
                (max(c.fill_color) - min(c.fill_color)) / max(c.fill_color)
                for c in all_skin_elements if c.fill_color[0] > c.fill_color[1] and c.fill_color[0] > c.fill_color[2] and max(c.fill_color) > 0
            ]
            if not all_red_saturations:
                return False
            
            saturation_threshold = np.percentile(all_red_saturations, 90)
            if saturation >= saturation_threshold:
                return True
        
        return False
    
    def get_element_importance_weights(self, all_skin_elements: List[SVGElement]) -> Dict[int, float]:
        """获取元素重要性权重（用于颜色量化）"""
        weights = {}
        
        # --- 修复核心：创建索引的集合，而非对象的集合 ---
        # 错误的做法: set(all_skin_elements)
        # 正确的做法: 创建一个包含所有皮肤元素索引的集合，以便快速查找
        skin_indices_set = {e.index for e in all_skin_elements}
        # --- 修复结束 ---

        for element in self.svg_parser.elements:
            weight = 1.0

            # --- 修复核心：检查元素的索引是否存在于集合中 ---
            if element.index in skin_indices_set:
                if self._is_likely_facial_feature(element, all_skin_elements):
                    weight = 3.0
            # --- 修复结束 ---
            
            coverage = self.coverage_ratios.get(element.index, 0)
            if 0.3 <= coverage <= 0.7:
                weight *= 1.5
            
            weights[element.index] = weight
        
        return weights

class FastSVGRegionMapper(SVGRegionMapper):
    """
    更快速的SVG区域映射器，使用基于边界框的近似方法（已修复）
    """
    
    def __init__(self, svg_parser: SVGParser, skin_mask: np.ndarray):
        super().__init__(svg_parser, skin_mask)
        self.use_approximation = True
        
    def _batch_calculate_coverage(self):
        """
        使用基于边界框的采样方法，快速且准确地计算覆盖率。
        此方法不再使用有缺陷的中心矩形猜测法。
        """
        print("使用快速边界框采样方法计算覆盖率...")
        
        # 获取小尺寸皮肤蒙版的尺寸
        h_small, w_small = self.skin_mask_small.shape
        
        # 获取原始SVG的尺寸
        svg_h = self.svg_parser.height
        svg_w = self.svg_parser.width

        # 防止除以零
        if svg_h == 0 or svg_w == 0:
            print("错误: SVG尺寸为零，无法计算映射。")
            for element in self.svg_parser.elements:
                self.coverage_ratios[element.index] = 0.0
            return

        # 计算从SVG坐标到小蒙版坐标的缩放比例
        x_scale = w_small / svg_w
        y_scale = h_small / svg_h
        
        # 遍历所有元素，计算其边界框在小蒙版上的覆盖率
        for element in self.svg_parser.elements:
            if not element.bbox or element.bbox[2] <= 0 or element.bbox[3] <= 0:
                self.coverage_ratios[element.index] = 0.0
                continue

            # 获取元素在SVG坐标系下的边界框 (xmin, ymin, width, height)
            xmin_svg, ymin_svg, w_svg, h_svg = element.bbox

            # 将SVG边界框坐标转换为小蒙版的像素坐标
            x_start = int(xmin_svg * x_scale)
            y_start = int(ymin_svg * y_scale)
            x_end = int((xmin_svg + w_svg) * x_scale)
            y_end = int((ymin_svg + h_svg) * y_scale)

            # 确保坐标在蒙版图像的有效范围内
            x_start = max(0, x_start)
            y_start = max(0, y_start)
            x_end = min(w_small, x_end)
            y_end = min(h_small, y_end)

            # 从小蒙版中提取出该边界框对应的区域 (Region of Interest)
            bbox_mask_roi = self.skin_mask_small[y_start:y_end, x_start:x_end]
            
            # 计算覆盖率
            if bbox_mask_roi.size > 0:
                skin_pixels_in_bbox = np.sum(bbox_mask_roi > 0)
                coverage = skin_pixels_in_bbox / bbox_mask_roi.size
            else:
                coverage = 0.0 # 如果边界框无效或太小，则覆盖率为0
                
            self.coverage_ratios[element.index] = coverage
            
        print("快速覆盖率计算完成")
    
    def _estimate_coverage_fast(self, element: SVGElement) -> float:
        # 这个函数在新的逻辑下不再被调用，但为了保持结构完整性，可以保留一个pass
        pass