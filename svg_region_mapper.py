# svg_region_mapper.py
"""
调试版SVG区域映射模块：增强面部特征检测
"""
import numpy as np
import cv2
from typing import List, Tuple, Dict, Set
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

def saturation(color: Tuple[int, int, int]) -> float:
    """计算颜色饱和度"""
    r, g, b = color
    r, g, b = r/255.0, g/255.0, b/255.0
    max_val = max(r, g, b)
    min_val = min(r, g, b)
    if max_val == 0:
        return 0
    return (max_val - min_val) / max_val

def color_contrast(color1: Tuple[int, int, int], color2: Tuple[int, int, int]) -> float:
    """计算两个颜色的对比度"""
    b1 = brightness(color1)
    b2 = brightness(color2)
    return abs(b1 - b2) / 255.0

class EnhancedSVGRegionMapper:
    """增强的SVG区域映射器 - 强化眼部和嘴部保护"""
    
    def __init__(self, svg_parser: SVGParser, skin_mask: np.ndarray):
        self.svg_parser = svg_parser
        self.skin_mask = skin_mask
        self.coverage_ratios: Dict[int, float] = {}
        
        mode = getattr(svg_config, 'PROCESSING_MODE', 'fast')
        config = svg_config.REGION_MAPPING_CONFIG.get(mode, svg_config.REGION_MAPPING_CONFIG['fast'])
        self.downsample_factor = config['downsample_factor']
        self.batch_size = config['batch_size']
        self.skin_mask_small = self._downsample_mask(skin_mask)
        
        # 面部特征保护参数
        self.facial_feature_colors: Set[Tuple[int, int, int]] = set()
        self.protected_elements: Dict[int, str] = {}  # element_index -> feature_type
        
    def _downsample_mask(self, mask: np.ndarray) -> np.ndarray:
        h, w = mask.shape
        new_h, new_w = h // self.downsample_factor, w // self.downsample_factor
        return cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        
    def map_regions(self, coverage_threshold: float = 0.5) -> Tuple[List[int], List[int], List[int], List[int]]:
        """增强的区域映射，强化面部特征保护"""
        skin_indices, initial_env_indices = [], []
        
        print("开始批量计算元素覆盖率...")
        self._batch_calculate_coverage()
        
        for element in self.svg_parser.elements:
            coverage = self.coverage_ratios.get(element.index, 0)
            if coverage >= coverage_threshold:
                skin_indices.append(element.index)
            else:
                initial_env_indices.append(element.index)

        print("使用增强算法识别眼睛和嘴部特征...")
        env_indices, eye_indices, mouth_indices = self._enhanced_facial_feature_detection(
            initial_env_indices, skin_indices
        )
        
        # 记录保护的特征
        for idx in eye_indices:
            self.protected_elements[idx] = "eye"
            if idx < len(self.svg_parser.elements):
                self.facial_feature_colors.add(self.svg_parser.elements[idx].fill_color)
        
        for idx in mouth_indices:
            self.protected_elements[idx] = "mouth"
            if idx < len(self.svg_parser.elements):
                self.facial_feature_colors.add(self.svg_parser.elements[idx].fill_color)
        
        print(f"增强识别完成：皮肤({len(skin_indices)}), 环境({len(env_indices)}), 眼睛({len(eye_indices)}), 嘴巴({len(mouth_indices)})")
        print(f"保护的面部特征颜色: {len(self.facial_feature_colors)} 种")
        
        return skin_indices, env_indices, eye_indices, mouth_indices

    def _enhanced_facial_feature_detection(self, non_skin_indices: List[int], skin_indices: List[int]) -> Tuple[List[int], List[int], List[int]]:
        """增强的面部特征检测算法 - 更宽松的检测条件"""
        
        # 1. 找到主要面部区域
        contours, _ = cv2.findContours(self.skin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return non_skin_indices, [], []
            
        main_face_contour = max(contours, key=cv2.contourArea)
        face_x, face_y, face_w, face_h = cv2.boundingRect(main_face_contour)
        
        print(f"  检测到面部区域: ({face_x}, {face_y}) 尺寸: {face_w}x{face_h}")
        
        # 2. 定义面部关键区域 - 扩大检测范围
        eye_region_y_min = face_y + face_h * 0.1  # 扩大到上部10%
        eye_region_y_max = face_y + face_h * 0.7  # 扩大到下部70%
        mouth_region_y_min = face_y + face_h * 0.5  # 从中部50%开始
        mouth_region_y_max = face_y + face_h * 0.95  # 扩大到下部95%
        
        # 3. 计算皮肤区域的平均颜色（用于对比）
        skin_colors = [self.svg_parser.elements[idx].fill_color for idx in skin_indices 
                      if idx < len(self.svg_parser.elements)]
        
        # 修复：确保有足够的皮肤颜色样本
        if not skin_colors:
            # 从所有元素中选择最可能的皮肤颜色
            all_colors = [elem.fill_color for elem in self.svg_parser.elements]
            skin_colors = [color for color in all_colors if self._is_likely_skin_color(color)]
        
        avg_skin_color = self._calculate_average_color(skin_colors) if skin_colors else (200, 170, 130)
        avg_skin_brightness = brightness(avg_skin_color)
        
        print(f"  平均皮肤颜色: RGB{avg_skin_color}, 亮度: {avg_skin_brightness:.1f}")
        
        # 4. 分析所有元素（不仅仅是非皮肤元素）
        eye_candidates = []
        mouth_candidates = []
        remaining_env = []
        
        # 检查所有元素，包括皮肤元素中可能的特征
        all_candidates = non_skin_indices + skin_indices
        
        for elem_idx in all_candidates:
            if elem_idx >= len(self.svg_parser.elements):
                remaining_env.append(elem_idx)
                continue
                
            element = self.svg_parser.elements[elem_idx]
            if not element.bbox:
                remaining_env.append(elem_idx)
                continue
            
            ex, ey, ew, eh = element.bbox
            elem_color = element.fill_color
            elem_brightness = brightness(elem_color)
            elem_saturation = saturation(elem_color)
            
            # 检查是否在面部区域内
            if not (face_x <= ex <= face_x + face_w and face_y <= ey <= face_y + face_h):
                if elem_idx in non_skin_indices:
                    remaining_env.append(elem_idx)
                continue
            
            print(f"  检查元素 {elem_idx}: 位置({ex}, {ey}), 颜色{elem_color}, 亮度{elem_brightness:.1f}")
            
            # 5. 眼部特征检测（更宽松的规则）
            if eye_region_y_min <= ey <= eye_region_y_max:
                is_eye_feature = self._is_eye_feature_relaxed(
                    elem_color, elem_brightness, elem_saturation, 
                    avg_skin_color, avg_skin_brightness, ew, eh
                )
                
                if is_eye_feature:
                    confidence = self._calculate_eye_confidence(
                        elem_color, elem_brightness, elem_saturation, ew, eh
                    )
                    eye_candidates.append((elem_idx, confidence))
                    print(f"    -> 眼部候选 (置信度: {confidence:.2f})")
                    continue
            
            # 6. 嘴部特征检测（更宽松的规则）
            if mouth_region_y_min <= ey <= mouth_region_y_max:
                is_mouth_feature = self._is_mouth_feature_relaxed(
                    elem_color, elem_brightness, elem_saturation,
                    avg_skin_color, avg_skin_brightness, ew, eh
                )
                
                if is_mouth_feature:
                    confidence = self._calculate_mouth_confidence(
                        elem_color, elem_brightness, elem_saturation, ew, eh
                    )
                    mouth_candidates.append((elem_idx, confidence))
                    print(f"    -> 嘴部候选 (置信度: {confidence:.2f})")
                    continue
            
            # 不是面部特征的元素
            if elem_idx in non_skin_indices:
                remaining_env.append(elem_idx)
        
        # 7. 选择最佳候选者 - 降低阈值
        eye_indices = self._select_best_features(eye_candidates, max_count=15, min_confidence=0.2)
        mouth_indices = self._select_best_features(mouth_candidates, max_count=8, min_confidence=0.2)
        
        print(f"  眼部候选: {len(eye_candidates)} -> 选择: {len(eye_indices)}")
        print(f"  嘴部候选: {len(mouth_candidates)} -> 选择: {len(mouth_indices)}")
        
        return remaining_env, eye_indices, mouth_indices

    def _is_likely_skin_color(self, color: Tuple[int, int, int]) -> bool:
        """判断是否可能是皮肤颜色"""
        r, g, b = color
        # 简单的皮肤色判断：偏红偏黄，饱和度适中
        return (r > g > b * 0.8 and 
                100 < r < 255 and 
                80 < g < 220 and 
                50 < b < 200)

    def _is_eye_feature_relaxed(self, color: Tuple[int, int, int], brightness_val: float, saturation_val: float,
                               avg_skin_color: Tuple[int, int, int], avg_skin_brightness: float,
                               width: float, height: float) -> bool:
        """更宽松的眼部特征判断"""
        
        r, g, b = color
        
        # 1. 白色/浅色检测（眼白） - 降低阈值
        if r > 200 and g > 200 and b > 200:
            return True
        
        # 2. 深色检测（瞳孔/眼线） - 提高阈值
        if brightness_val < 80:
            return True
        
        # 3. 彩色检测（虹膜） - 降低要求
        if saturation_val > 0.2 and 30 < brightness_val < 180:
            contrast = color_contrast(color, avg_skin_color)
            if contrast > 0.1:  # 降低对比度要求
                return True
        
        # 4. 中等亮度但与皮肤对比明显
        contrast = color_contrast(color, avg_skin_color)
        if contrast > 0.15:  # 降低对比度要求
            return True
        
        # 5. 特殊颜色（蓝色、绿色、棕色眼睛）
        if ((b > r and b > g) or  # 蓝色倾向
            (g > r and g > b) or  # 绿色倾向
            (r > 100 and g > 80 and b < 100)):  # 棕色倾向
            return True
        
        return False

    def _is_mouth_feature_relaxed(self, color: Tuple[int, int, int], brightness_val: float, saturation_val: float,
                                 avg_skin_color: Tuple[int, int, int], avg_skin_brightness: float,
                                 width: float, height: float) -> bool:
        """更宽松的嘴部特征判断"""
        
        r, g, b = color
        
        # 1. 白色/浅色检测（牙齿） - 降低阈值
        if r > 200 and g > 200 and b > 200:
            return True
        
        # 2. 红色倾向检测（嘴唇） - 降低要求
        if r > g + 10 and r > b + 5:  # 降低红色要求
            return True
        
        # 3. 饱和度检测（唇彩/口红） - 降低要求
        if saturation_val > 0.25:  # 降低饱和度要求
            return True
        
        # 4. 深色检测（嘴巴内部）
        if brightness_val < 100:  # 提高深色阈值
            return True
        
        # 5. 与皮肤对比明显
        contrast = color_contrast(color, avg_skin_color)
        if contrast > 0.1:  # 降低对比度要求
            return True
        
        return False

    def _calculate_eye_confidence(self, color: Tuple[int, int, int], brightness_val: float, 
                                 saturation_val: float, width: float, height: float) -> float:
        """计算眼部特征的置信度"""
        confidence = 0.0
        r, g, b = color
        
        # 白色眼白 - 最高置信度
        if r > 200 and g > 200 and b > 200:
            confidence += 0.9
        
        # 深色瞳孔 - 高置信度
        elif brightness_val < 80:
            confidence += 0.8
        
        # 饱和颜色（彩色眼睛）
        if saturation_val > 0.2:
            confidence += 0.5
        
        # 尺寸合理性
        if width < 100 and height < 100:  # 放宽尺寸限制
            confidence += 0.3
        
        return min(confidence, 1.0)  # 限制最大值为1.0

    def _calculate_mouth_confidence(self, color: Tuple[int, int, int], brightness_val: float,
                                   saturation_val: float, width: float, height: float) -> float:
        """计算嘴部特征的置信度"""
        confidence = 0.0
        r, g, b = color
        
        # 白色牙齿 - 高置信度
        if r > 200 and g > 200 and b > 200:
            confidence += 0.8
        
        # 红色嘴唇
        if r > g + 10 and r > b + 5:
            confidence += 0.6
        
        # 饱和度（口红等）
        if saturation_val > 0.25:
            confidence += 0.7
        
        # 深色（嘴巴内部）
        if brightness_val < 100:
            confidence += 0.5
        
        return min(confidence, 1.0)  # 限制最大值为1.0

    def _select_best_features(self, candidates: List[Tuple[int, float]], max_count: int, min_confidence: float = 0.3) -> List[int]:
        """选择最佳的面部特征元素"""
        if not candidates:
            return []
        
        # 按置信度排序
        candidates.sort(key=lambda x: x[1], reverse=True)
        
        # 选择前N个高置信度的特征
        selected = [idx for idx, confidence in candidates[:max_count] if confidence >= min_confidence]
        
        return selected

    def _calculate_average_color(self, colors: List[Tuple[int, int, int]]) -> Tuple[int, int, int]:
        """计算颜色列表的平均值"""
        if not colors:
            return (200, 170, 130)  # 默认肤色
        
        avg_r = sum(c[0] for c in colors) / len(colors)
        avg_g = sum(c[1] for c in colors) / len(colors)
        avg_b = sum(c[2] for c in colors) / len(colors)
        
        return (int(avg_r), int(avg_g), int(avg_b))

    def get_element_importance_weights(self, skin_indices, eye_indices, mouth_indices) -> Dict[int, float]:
        """生成强化的重要性权重"""
        print("生成强化的面部特征保护权重...")
        weights = {}
        skin_elements = [e for e in self.svg_parser.elements if e.index in skin_indices]
        
        # 1. 默认权重
        for element in self.svg_parser.elements:
            weights[element.index] = 1.0

        # 2. 边界权重
        for idx, coverage in self.coverage_ratios.items():
            if 0.3 <= coverage <= 0.7:
                weights[idx] = 2.0

        # 3. 重要皮肤特征权重
        if skin_elements:
            all_brightness = [brightness(e.fill_color) for e in skin_elements]
            if all_brightness:
                brightness_threshold = np.percentile(all_brightness, 10)
                for elem in skin_elements:
                    if brightness(elem.fill_color) < brightness_threshold:
                        weights[elem.index] = 8.0

        # 4. ⭐ 最高优先级：眼睛和嘴巴（大幅提升权重）
        for idx in eye_indices:
            weights[idx] = 100.0  # 超高权重确保不被合并
            
        for idx in mouth_indices:
            weights[idx] = 100.0  # 超高权重确保不被合并
        
        print(f"  设置了 {len(eye_indices)} 个眼部元素的超高权重")
        print(f"  设置了 {len(mouth_indices)} 个嘴部元素的超高权重")
        
        return weights

    def get_protected_colors(self) -> Set[Tuple[int, int, int]]:
        """获取需要强制保护的颜色"""
        return self.facial_feature_colors.copy()

    def get_protected_elements(self) -> Dict[int, str]:
        """获取受保护的元素映射"""
        return self.protected_elements.copy()

    def _batch_calculate_coverage(self):
        """批量计算覆盖率"""
        elements_list = list(self.svg_parser.elements)
        for i in range(0, len(elements_list), self.batch_size):
            batch = elements_list[i:i + self.batch_size]
            self._process_batch_color_method(batch)
            print(f"  进度: {min(100, (i + self.batch_size) * 100 // len(elements_list))}%", end='\r')
        print("\n覆盖率计算完成")

    def _process_batch_color_method(self, batch: List[SVGElement]):
        """批量处理方法"""
        import cairosvg
        from io import BytesIO
        from PIL import Image
        
        h, w = self.skin_mask_small.shape
        svg = ET.Element('svg', {
            'xmlns': 'http://www.w3.org/2000/svg', 
            'width': str(self.svg_parser.width), 
            'height': str(self.svg_parser.height), 
            'viewBox': f'0 0 {self.svg_parser.width} {self.svg_parser.height}'
        })
        
        # 白色背景
        ET.SubElement(svg, 'rect', {
            'x': '0', 'y': '0', 
            'width': str(self.svg_parser.width), 
            'height': str(self.svg_parser.height), 
            'fill': 'white'
        })
        
        color_to_idx = {}
        for i, elem in enumerate(batch):
            color_idx = i + 1
            r, g, b = (color_idx >> 16) & 0xFF, (color_idx >> 8) & 0xFF, color_idx & 0xFF
            path = ET.SubElement(svg, 'path', {
                'd': elem.path_data, 
                'fill': f'#{r:02x}{g:02x}{b:02x}'
            })
            if elem.transform:
                path.set('transform', elem.transform)
            color_to_idx[color_idx] = elem.index
        
        try:
            h_orig, w_orig = self.skin_mask.shape
            png_data = cairosvg.svg2png(
                bytestring=ET.tostring(svg), 
                output_width=w_orig, 
                output_height=h_orig
            )
            rendered_rgb = cv2.resize(
                np.array(Image.open(BytesIO(png_data)).convert('RGB')), 
                (w, h), 
                interpolation=cv2.INTER_NEAREST
            )
            
            for color_idx, elem_idx in color_to_idx.items():
                r, g, b = (color_idx >> 16) & 0xFF, (color_idx >> 8) & 0xFF, color_idx & 0xFF
                color_mask = np.all(rendered_rgb == [r, g, b], axis=-1)
                element_area = np.sum(color_mask)
                
                if element_area > 0:
                    skin_overlap = np.sum(color_mask & (self.skin_mask_small > 0))
                    self.coverage_ratios[elem_idx] = skin_overlap / element_area
                else:
                    self.coverage_ratios[elem_idx] = 0.0
                    
        except Exception as e:
            print(f"批量渲染失败: {e}")
            for elem in batch:
                self.coverage_ratios.setdefault(elem.index, 0.5)

# 保持向后兼容
SVGRegionMapper = EnhancedSVGRegionMapper

class FastSVGRegionMapper(EnhancedSVGRegionMapper):
    """快速版本的增强区域映射器"""
    
    def _batch_calculate_coverage(self):
        """使用快速边界框方法"""
        print("使用快速边界框采样方法计算覆盖率...")
        h_small, w_small = self.skin_mask_small.shape
        svg_h, svg_w = self.svg_parser.height, self.svg_parser.width
        
        if svg_h == 0 or svg_w == 0:
            for e in self.svg_parser.elements:
                self.coverage_ratios.setdefault(e.index, 0.0)
            return
        
        x_scale, y_scale = w_small / svg_w, h_small / svg_h
        
        for element in self.svg_parser.elements:
            if not element.bbox or element.bbox[2] <= 0 or element.bbox[3] <= 0:
                self.coverage_ratios[element.index] = 0.0
                continue
            
            xmin_svg, ymin_svg, w_svg, h_svg = element.bbox
            x_start, y_start = int(xmin_svg * x_scale), int(ymin_svg * y_scale)
            x_end, y_end = int((xmin_svg + w_svg) * x_scale), int((ymin_svg + h_svg) * y_scale)
            
            x_start = max(0, x_start)
            y_start = max(0, y_start)
            x_end = min(w_small, x_end)
            y_end = min(h_small, y_end)
            
            roi = self.skin_mask_small[y_start:y_end, x_start:x_end]
            if roi.size > 0:
                self.coverage_ratios[element.index] = np.sum(roi > 0) / roi.size
            else:
                self.coverage_ratios[element.index] = 0.0
        
        print("快速覆盖率计算完成")