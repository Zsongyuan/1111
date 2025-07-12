# svg_color_quantizer.py
"""
改进的保护面部特征的SVG颜色量化模块
"""
import numpy as np
import torch
from typing import List, Tuple, Dict, Optional, Set
import cv2

class ProtectedSVGColorQuantizer:
    """保护面部特征的SVG颜色量化器"""
    
    def __init__(self, use_gpu: bool = True):
        self.device = torch.device('cuda' if use_gpu and torch.cuda.is_available() else 'cpu')
        print(f"颜色量化器使用设备: {self.device}")
        
    def quantize_colors(self, 
                       element_colors: List[Tuple[int, int, int]], 
                       k: int,
                       weights: Optional[List[float]] = None,
                       protected_colors: Optional[Set[Tuple[int, int, int]]] = None,
                       saturation_factor: float = 1.2) -> Tuple[List[int], List[Tuple[int, int, int]]]:
        """改进的保护重要颜色的量化算法"""
        
        if not element_colors or k <= 0:
            return [], []
        
        # 增强饱和度
        enhanced_colors = self._enhance_saturation(element_colors, saturation_factor)
        protected_colors = protected_colors or set()
        
        print(f"  量化参数: 输入颜色={len(element_colors)}, 目标={k}, 保护颜色={len(protected_colors)}")
        
        # 标准化保护颜色格式
        normalized_protected = set()
        for color in protected_colors:
            if isinstance(color, (list, tuple)) and len(color) >= 3:
                normalized_protected.add(tuple(int(c) for c in color[:3]))
        
        # 标准化输入颜色并找到真正存在的保护颜色
        normalized_enhanced = []
        for color in enhanced_colors:
            if isinstance(color, (list, tuple)) and len(color) >= 3:
                normalized_enhanced.append(tuple(int(c) for c in color[:3]))
            else:
                normalized_enhanced.append(color)
        
        # 找到输入中真正存在的保护颜色 - 使用更宽松的匹配
        input_color_set = set(normalized_enhanced)
        actual_protected = set()
        
        for protected_color in normalized_protected:
            # 首先尝试精确匹配
            if protected_color in input_color_set:
                actual_protected.add(protected_color)
            else:
                # 如果精确匹配失败，寻找最相似的颜色
                closest_color = self._find_closest_color(protected_color, list(input_color_set))
                if closest_color and self._safe_color_distance(protected_color, closest_color) < 30:  # 允许30的颜色距离
                    actual_protected.add(closest_color)
        
        # 调试输出
        print(f"  保护颜色匹配详情:")
        for protected_color in normalized_protected:
            if protected_color in actual_protected:
                print(f"    {protected_color}: ✓ 直接匹配")
            else:
                closest = self._find_closest_color(protected_color, list(input_color_set))
                distance = self._safe_color_distance(protected_color, closest) if closest else float('inf')
                print(f"    {protected_color}: ✗ 最近颜色 {closest}, 距离 {distance:.1f}")
        
        print(f"  实际保护的颜色: {len(actual_protected)} 种")
        
        # 如果没有实际的保护颜色，使用标准K-means
        if not actual_protected:
            return self._standard_kmeans(normalized_enhanced, k, weights)
        
        # 如果保护颜色数量 >= k，直接返回保护颜色为主的结果
        if len(actual_protected) >= k:
            protected_list = list(actual_protected)[:k]
            labels = self._assign_labels_to_protected(normalized_enhanced, protected_list)
            return labels, protected_list
        
        # 混合策略：保护颜色 + 量化其他颜色
        return self._hybrid_quantization(normalized_enhanced, k, actual_protected, weights)

    def _hybrid_quantization(self, colors: List[Tuple[int, int, int]], k: int, 
                           protected_colors: Set[Tuple[int, int, int]], 
                           weights: Optional[List[float]]) -> Tuple[List[int], List[Tuple[int, int, int]]]:
        """混合量化策略：保护颜色 + K-means量化"""
        
        # 分离保护颜色和普通颜色
        protected_indices = []
        normal_indices = []
        
        for i, color in enumerate(colors):
            if color in protected_colors:
                protected_indices.append(i)
            else:
                normal_indices.append(i)
        
        # 保护颜色直接保留
        protected_list = list(protected_colors)
        
        # 计算普通颜色需要的聚类数
        k_normal = max(0, k - len(protected_list))
        
        if k_normal > 0 and normal_indices:
            # 对普通颜色进行K-means
            normal_colors = [colors[i] for i in normal_indices]
            normal_weights = [weights[i] if weights else 1.0 for i in normal_indices] if weights else None
            
            # 去除重复颜色
            unique_normal = list(set(normal_colors))
            
            if len(unique_normal) <= k_normal:
                normal_centroids = unique_normal
            else:
                # 使用K-means量化
                if self.device.type == 'cuda':
                    try:
                        normal_centroids = self._gpu_kmeans(normal_colors, k_normal, normal_weights)
                    except:
                        normal_centroids = self._cpu_kmeans(normal_colors, k_normal, normal_weights)
                else:
                    normal_centroids = self._cpu_kmeans(normal_colors, k_normal, normal_weights)
        else:
            normal_centroids = []
        
        # 合并所有聚类中心
        all_centroids = protected_list + normal_centroids
        
        # 为每个颜色分配标签
        labels = []
        for color in colors:
            if color in protected_colors:
                # 直接分配到对应的保护颜色
                labels.append(protected_list.index(color))
            else:
                # 找到最近的聚类中心
                distances = [self._safe_color_distance(color, centroid) for centroid in all_centroids]
                labels.append(np.argmin(distances))
        
        print(f"  量化结果: {len(all_centroids)} 种颜色 (保护:{len(protected_list)}, 量化:{len(normal_centroids)})")
        
        return labels, all_centroids

    def _standard_kmeans(self, colors: List[Tuple[int, int, int]], k: int, 
                        weights: Optional[List[float]]) -> Tuple[List[int], List[Tuple[int, int, int]]]:
        """标准K-means量化"""
        unique_colors = list(set(colors))
        
        if len(unique_colors) <= k:
            # 直接返回唯一颜色
            color_to_label = {color: i for i, color in enumerate(unique_colors)}
            labels = [color_to_label[color] for color in colors]
            return labels, unique_colors
        
        # 使用K-means
        if self.device.type == 'cuda':
            try:
                centroids = self._gpu_kmeans(colors, k, weights)
            except:
                centroids = self._cpu_kmeans(colors, k, weights)
        else:
            centroids = self._cpu_kmeans(colors, k, weights)
        
        # 分配标签
        labels = []
        for color in colors:
            distances = [self._safe_color_distance(color, centroid) for centroid in centroids]
            labels.append(np.argmin(distances))
        
        return labels, centroids

    def _assign_labels_to_protected(self, colors: List[Tuple[int, int, int]], 
                                   protected_list: List[Tuple[int, int, int]]) -> List[int]:
        """将颜色分配给保护颜色"""
        labels = []
        for color in colors:
            if color in protected_list:
                labels.append(protected_list.index(color))
            else:
                # 找最近的保护颜色
                distances = [self._safe_color_distance(color, pc) for pc in protected_list]
                labels.append(np.argmin(distances))
        return labels

    def _find_closest_color(self, target_color: Tuple[int, int, int], 
                          color_list: List[Tuple[int, int, int]]) -> Tuple[int, int, int]:
        """找到最接近的颜色"""
        if not color_list:
            return target_color
        
        min_distance = float('inf')
        closest_color = color_list[0]
        
        for color in color_list:
            distance = self._safe_color_distance(target_color, color)
            if distance < min_distance:
                min_distance = distance
                closest_color = color
        
        return closest_color

    def _safe_color_distance(self, color1: Tuple[int, int, int], color2: Tuple[int, int, int]) -> float:
        """安全的颜色距离计算，避免overflow"""
        try:
            # 确保输入是有效的
            c1 = tuple(int(x) for x in color1[:3])
            c2 = tuple(int(x) for x in color2[:3])
            
            # 使用float避免overflow
            distance = sum((float(a) - float(b)) ** 2 for a, b in zip(c1, c2))
            return float(np.sqrt(distance))
        except:
            return float('inf')

    def _cpu_kmeans(self, colors: List[Tuple[int, int, int]], k: int, 
                   weights: Optional[List[float]]) -> List[Tuple[int, int, int]]:
        """CPU版本的K-means"""
        colors_array = np.array(colors, dtype=np.float32)
        weights_array = np.array(weights if weights else np.ones(len(colors)), dtype=np.float32)
        
        # K-means++初始化
        centroids = self._kmeans_plusplus_init_cpu(colors_array, k, weights_array)
        
        # 迭代优化
        for iteration in range(50):  # 减少迭代次数
            # 分配标签
            distances = np.array([[self._safe_color_distance(color, centroid) 
                                 for centroid in centroids] for color in colors])
            labels = np.argmin(distances, axis=1)
            
            # 更新聚类中心
            new_centroids = []
            for i in range(k):
                mask = (labels == i)
                if np.any(mask):
                    masked_colors = colors_array[mask]
                    masked_weights = weights_array[mask]
                    weighted_mean = np.average(masked_colors, axis=0, weights=masked_weights)
                    new_centroids.append(tuple(int(x) for x in weighted_mean))
                else:
                    new_centroids.append(centroids[i])
            
            # 检查收敛
            if np.allclose([list(c) for c in centroids], [list(c) for c in new_centroids], atol=1.0):
                break
            
            centroids = new_centroids
        
        return centroids

    def _gpu_kmeans(self, colors: List[Tuple[int, int, int]], k: int, 
                   weights: Optional[List[float]]) -> List[Tuple[int, int, int]]:
        """GPU版本的K-means"""
        colors_array = np.array(colors, dtype=np.float32)
        weights_array = np.array(weights if weights else np.ones(len(colors)))
        
        labels, centroids_np = self._weighted_kmeans_gpu(colors_array, k, weights_array)
        centroids = [(int(round(c[0])), int(round(c[1])), int(round(c[2]))) for c in centroids_np]
        return centroids

    def _kmeans_plusplus_init_cpu(self, colors: np.ndarray, k: int, weights: np.ndarray) -> List[Tuple[int, int, int]]:
        """CPU版K-means++初始化"""
        n_samples = len(colors)
        centroids = []
        
        # 选择第一个中心
        probs = weights / weights.sum()
        first_idx = np.random.choice(n_samples, p=probs)
        centroids.append(tuple(int(x) for x in colors[first_idx]))
        
        # 选择剩余中心
        for i in range(1, k):
            distances = np.array([min(self._safe_color_distance(color, c) for c in centroids) 
                                for color in colors])
            probs = (distances ** 2) * weights
            probs /= probs.sum()
            
            if np.sum(probs) > 0:
                next_idx = np.random.choice(n_samples, p=probs)
                centroids.append(tuple(int(x) for x in colors[next_idx]))
            else:
                # 随机选择
                next_idx = np.random.choice(n_samples)
                centroids.append(tuple(int(x) for x in colors[next_idx]))
        
        return centroids

    def _enhance_saturation(self, colors, factor):
        """增强饱和度"""
        enhanced = []
        for r, g, b in colors:
            try:
                rgb = np.uint8([[[r, g, b]]])
                hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
                hsv[0, 0, 1] = np.clip(hsv[0, 0, 1] * factor, 0, 255)
                rgb_enhanced = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
                enhanced.append(tuple(rgb_enhanced[0, 0]))
            except:
                enhanced.append((r, g, b))  # 失败时返回原色
        return enhanced

    def _weighted_kmeans_gpu(self, colors: np.ndarray, k: int, weights: np.ndarray, 
                            max_iters: int = 50, tol: float = 1e-4) -> Tuple[np.ndarray, np.ndarray]:
        """GPU加速的加权K-means"""
        colors_gpu = torch.from_numpy(colors).to(self.device)
        weights_gpu = torch.from_numpy(weights).float().to(self.device)
        centroids_gpu = self._kmeans_plusplus_init(colors_gpu, k, weights_gpu)
        
        for iteration in range(max_iters):
            distances = torch.cdist(colors_gpu, centroids_gpu)
            prev_labels = labels if 'labels' in locals() else None
            labels = torch.argmin(distances, dim=1)
            
            if prev_labels is not None and torch.all(labels == prev_labels):
                break
            
            new_centroids = torch.zeros_like(centroids_gpu)
            for i in range(k):
                mask = (labels == i)
                if torch.any(mask):
                    masked_colors = colors_gpu[mask]
                    masked_weights = weights_gpu[mask]
                    weighted_sum = torch.sum(masked_colors * masked_weights.unsqueeze(1), dim=0)
                    weight_sum = torch.sum(masked_weights)
                    new_centroids[i] = weighted_sum / weight_sum
                else:
                    new_centroids[i] = centroids_gpu[i]
            
            if torch.max(torch.norm(new_centroids - centroids_gpu, dim=1)) < tol:
                break
            centroids_gpu = new_centroids
        
        return labels.cpu().numpy(), centroids_gpu.cpu().numpy()

    def _kmeans_plusplus_init(self, colors: torch.Tensor, k: int, weights: torch.Tensor) -> torch.Tensor:
        """K-means++初始化"""
        n_samples = colors.shape[0]
        centroids = torch.zeros((k, 3), device=self.device)
        
        # 选择第一个中心
        probs = weights / weights.sum()
        first_idx = torch.multinomial(probs, 1).item()
        centroids[0] = colors[first_idx]
        
        # 选择剩余中心
        for i in range(1, k):
            min_distances, _ = torch.min(torch.cdist(colors, centroids[:i]), dim=1)
            probs = (min_distances ** 2) * weights
            probs /= probs.sum()
            next_idx = torch.multinomial(probs, 1).item()
            centroids[i] = colors[next_idx]
        
        return centroids

class EnhancedRegionAwareSVGQuantizer:
    """增强的区域感知SVG量化器 - 保护面部特征"""
    
    def __init__(self, use_gpu: bool = True):
        self.quantizer = ProtectedSVGColorQuantizer(use_gpu)
        
    def quantize_by_regions(self,
                           facial_elements: List[Tuple[int, Tuple[int, int, int]]],
                           env_elements: List[Tuple[int, Tuple[int, int, int]]],
                           k_facial: int,
                           k_env: int,
                           importance_weights: Dict[int, float],
                           protected_colors: Optional[Set[Tuple[int, int, int]]] = None,
                           saturation_factor: float = 1.2) -> Dict[int, Tuple[int, int, int]]:
        """增强的区域量化 - 强制保护重要颜色"""
        
        color_mapping = {}
        protected_colors = protected_colors or set()
        
        print(f"开始保护性区域量化...")
        print(f"  面部元素: {len(facial_elements)}, 环境元素: {len(env_elements)}")
        print(f"  保护颜色: {len(protected_colors)} 种")

        # 处理面部区域（包含保护颜色）
        if facial_elements and k_facial > 0:
            facial_indices = [idx for idx, _ in facial_elements]
            facial_colors = [color for _, color in facial_elements]
            facial_weights = [importance_weights.get(idx, 1.0) for idx in facial_indices]
            
            # 检查面部区域的保护颜色
            facial_protected = set()
            for color in facial_colors:
                if color in protected_colors:
                    facial_protected.add(color)
            
            print(f"  面部区域保护颜色: {len(facial_protected)} 种")
            
            labels, centroids = self.quantizer.quantize_colors(
                facial_colors, k_facial, facial_weights, facial_protected, saturation_factor
            )
            
            if labels is not None and len(labels) > 0 and len(centroids) > 0:
                for i, idx in enumerate(facial_indices):
                    if i < len(labels) and labels[i] < len(centroids):
                        color_mapping[idx] = centroids[labels[i]]

        # 处理环境区域
        if env_elements and k_env > 0:
            env_indices = [idx for idx, _ in env_elements]
            env_colors = [color for _, color in env_elements]
            env_weights = [importance_weights.get(idx, 1.0) for idx in env_indices]
            
            labels, centroids = self.quantizer.quantize_colors(
                env_colors, k_env, env_weights, set(), saturation_factor  # 环境区域不使用保护颜色
            )

            if labels is not None and len(labels) > 0 and len(centroids) > 0:
                for i, idx in enumerate(env_indices):
                    if i < len(labels) and labels[i] < len(centroids):
                        color_mapping[idx] = centroids[labels[i]]
        
        # 验证保护颜色是否被保留
        final_colors = set(color_mapping.values())
        preserved_protected = protected_colors & final_colors
        print(f"  成功保留的保护颜色: {len(preserved_protected)}/{len(protected_colors)}")
        
        if len(preserved_protected) < len(protected_colors):
            missing_colors = protected_colors - preserved_protected
            print(f"  警告: 以下保护颜色可能丢失: {list(missing_colors)[:3]}...")
        
        return color_mapping

# 保持向后兼容
SVGColorQuantizer = ProtectedSVGColorQuantizer
RegionAwareSVGQuantizer = EnhancedRegionAwareSVGQuantizer