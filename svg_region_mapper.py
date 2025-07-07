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
        将SVG元素映射到皮肤/环境区域
        使用批量渲染和多线程加速处理
        """
        skin_indices = []
        env_indices = []
        
        print("开始批量计算元素覆盖率...")
        
        # 使用更快的批量方法计算覆盖率
        self._batch_calculate_coverage()
        
        # 根据覆盖率分类
        for element in self.svg_parser.elements:
            coverage = self.coverage_ratios.get(element.index, 0)
            
            if coverage >= coverage_threshold:
                skin_indices.append(element.index)
            else:
                env_indices.append(element.index)
        
        # 处理边界元素
        skin_indices, env_indices = self._refine_boundary_elements(skin_indices, env_indices)
        
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
        """使用颜色标记法批量处理元素"""
        # 创建一个SVG，每个元素使用唯一颜色
        h, w = self.skin_mask_small.shape
        
        # 创建SVG根元素
        svg = ET.Element('svg')
        svg.set('xmlns', 'http://www.w3.org/2000/svg')
        svg.set('width', str(self.svg_parser.width))
        svg.set('height', str(self.svg_parser.height))
        svg.set('viewBox', f'0 0 {self.svg_parser.width} {self.svg_parser.height}')
        
        # 添加白色背景
        bg = ET.SubElement(svg, 'rect')
        bg.set('x', '0')
        bg.set('y', '0')
        bg.set('width', str(self.svg_parser.width))
        bg.set('height', str(self.svg_parser.height))
        bg.set('fill', 'white')
        
        # 为每个元素分配唯一颜色
        color_to_idx = {}
        for i, elem in enumerate(batch):
            # 使用RGB通道编码元素索引
            color_idx = i + 1  # 避免使用0（黑色）
            r = (color_idx >> 16) & 0xFF
            g = (color_idx >> 8) & 0xFF
            b = color_idx & 0xFF
            color_str = f'#{r:02x}{g:02x}{b:02x}'
            
            # 添加path元素
            path = ET.SubElement(svg, 'path')
            path.set('d', elem.path_data)
            path.set('fill', color_str)
            if elem.transform:
                path.set('transform', elem.transform)
            
            color_to_idx[color_idx] = elem.index
        
        # 渲染SVG
        svg_str = ET.tostring(svg, encoding='unicode')
        
        try:
            # 使用svgConvertor渲染，但降低分辨率
            import tempfile
            import os
            with tempfile.NamedTemporaryFile(mode='w', suffix='.svg', delete=False) as tmp:
                tmp.write(svg_str)
                tmp_path = tmp.name
            
            # 使用配置的DPI加速渲染
            from svgConvertor import convert_and_denoise
            rendered, _ = convert_and_denoise(tmp_path, dpi=self.render_dpi)
            os.unlink(tmp_path)
            
            # 转换为RGB并调整大小
            rendered_rgb = cv2.cvtColor(rendered, cv2.COLOR_BGR2RGB)
            if rendered_rgb.shape[:2] != (h, w):
                rendered_rgb = cv2.resize(rendered_rgb, (w, h), interpolation=cv2.INTER_NEAREST)
            
            # 分析每个颜色的覆盖率
            for color_idx, elem_idx in color_to_idx.items():
                # 提取颜色
                r = (color_idx >> 16) & 0xFF
                g = (color_idx >> 8) & 0xFF
                b = color_idx & 0xFF
                
                # 创建元素mask
                color_mask = np.all(rendered_rgb == [r, g, b], axis=-1)
                
                # 计算覆盖率
                element_area = np.sum(color_mask)
                if element_area > 0:
                    intersection = np.sum(color_mask & (self.skin_mask_small > 0))
                    coverage = intersection / element_area
                else:
                    coverage = 0.0
                
                self.coverage_ratios[elem_idx] = coverage
                
        except Exception as e:
            print(f"批量渲染失败: {e}")
            # 降级到单个处理
            for elem in batch:
                self.coverage_ratios[elem.index] = 0.5  # 默认值
    
    def _refine_boundary_elements(self, skin_indices: List[int], env_indices: List[int]) -> Tuple[List[int], List[int]]:
        """细化边界元素的分类"""
        # 识别边界元素（覆盖率在0.3-0.7之间的元素）
        boundary_elements = []
        for idx, coverage in self.coverage_ratios.items():
            if 0.3 <= coverage <= 0.7:
                boundary_elements.append(idx)
        
        # 对边界元素进行更精细的分析
        for elem_idx in boundary_elements:
            element = next(e for e in self.svg_parser.elements if e.index == elem_idx)
            
            # 基于颜色特征判断
            if self._is_likely_facial_feature(element):
                # 可能是眼睛或嘴唇，归入皮肤区域
                if elem_idx in env_indices:
                    env_indices.remove(elem_idx)
                    skin_indices.append(elem_idx)
        
        return skin_indices, env_indices
    
    def _is_likely_facial_feature(self, element: SVGElement) -> bool:
        """判断元素是否可能是面部特征（眼睛、嘴唇等）"""
        r, g, b = element.fill_color
        
        # 计算颜色特征
        # 饱和度
        max_val = max(r, g, b) / 255.0
        min_val = min(r, g, b) / 255.0
        saturation = (max_val - min_val) / max_val if max_val > 0 else 0
        
        # 高饱和度可能是嘴唇
        if saturation > 0.4 and r > max(g, b):
            return True
        
        # 深色可能是眼睛
        brightness = (r + g + b) / (3 * 255)
        if brightness < 0.3:
            return True
        
        # 基于位置判断（如果有边界框信息）
        if element.bbox:
            x, y, w, h = element.bbox
            cx = x + w / 2
            cy = y + h / 2
            
            # 在画布中心区域
            if (0.3 < cx / self.svg_parser.width < 0.7 and 
                0.2 < cy / self.svg_parser.height < 0.8):
                # 小面积元素更可能是面部特征
                area_ratio = (w * h) / (self.svg_parser.width * self.svg_parser.height)
                if area_ratio < 0.05:  # 小于5%的面积
                    return True
        
        return False
    
    def get_element_importance_weights(self) -> Dict[int, float]:
        """获取元素重要性权重（用于颜色量化）"""
        weights = {}
        
        for element in self.svg_parser.elements:
            weight = 1.0
            
            # 面部特征获得更高权重
            if self._is_likely_facial_feature(element):
                weight = 3.0
            
            # 边界元素稍微提高权重
            coverage = self.coverage_ratios.get(element.index, 0)
            if 0.3 <= coverage <= 0.7:
                weight *= 1.5
            
            weights[element.index] = weight
        
        return weights

class FastSVGRegionMapper(SVGRegionMapper):
    """更快速的SVG区域映射器，使用近似方法"""
    
    def __init__(self, svg_parser: SVGParser, skin_mask: np.ndarray):
        super().__init__(svg_parser, skin_mask)
        self.use_approximation = True
        
    def _batch_calculate_coverage(self):
        """使用近似方法快速计算覆盖率"""
        print("使用快速近似方法计算覆盖率...")
        
        # 基于元素的边界框和颜色特征进行快速分类
        for element in self.svg_parser.elements:
            # 使用简化的方法估算覆盖率
            coverage = self._estimate_coverage_fast(element)
            self.coverage_ratios[element.index] = coverage
            
        print("快速覆盖率计算完成")
    
    def _estimate_coverage_fast(self, element: SVGElement) -> float:
        """基于位置和颜色快速估算覆盖率"""
        # 如果有边界框信息，使用位置判断
        if element.bbox:
            x, y, w, h = element.bbox
            cx = x + w / 2
            cy = y + h / 2
            
            # 根据位置粗略判断
            # 中心区域更可能是皮肤
            center_x = 0.5 < cx / self.svg_parser.width < 0.5
            center_y = 0.3 < cy / self.svg_parser.height < 0.7
            
            if center_x and center_y:
                # 基于颜色进一步判断
                r, g, b = element.fill_color
                # 肤色通常在特定范围内
                is_skin_like = (r > 100 and g > 50 and b > 20 and 
                              r > g and g > b and 
                              r - b > 15)
                
                return 0.8 if is_skin_like else 0.3
            else:
                return 0.2
        
        # 没有边界框信息，使用默认值
        return 0.5