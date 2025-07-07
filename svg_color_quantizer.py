# svg_color_quantizer.py
"""
SVG颜色量化模块：使用GPU加速对SVG元素的颜色进行聚类和量化
"""
import numpy as np
import torch
import torch.nn as nn
from typing import List, Tuple, Dict, Optional
from sklearn.cluster import KMeans
import cv2

class SVGColorQuantizer:
    """基于GPU的SVG颜色量化器"""
    
    def __init__(self, use_gpu: bool = True):
        self.device = torch.device('cuda' if use_gpu and torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {self.device}")
        
    def quantize_colors(self, 
                       element_colors: List[Tuple[int, int, int]], 
                       k: int,
                       weights: Optional[List[float]] = None,
                       saturation_factor: float = 1.2) -> Tuple[List[int], List[Tuple[int, int, int]]]:
        """
        对SVG元素颜色进行量化
        
        参数:
            element_colors: 元素颜色列表 [(R, G, B), ...]
            k: 目标颜色数
            weights: 元素权重列表（用于保护重要元素）
            saturation_factor: 饱和度增强系数
            
        返回:
            (labels, centroids): 聚类标签和聚类中心颜色
        """
        if len(element_colors) == 0:
            return [], []
        
        # 预处理：增强饱和度
        enhanced_colors = self._enhance_saturation(element_colors, saturation_factor)
        
        # 如果颜色数少于k，直接返回
        unique_colors = list(set(enhanced_colors))
        if len(unique_colors) <= k:
            color_to_label = {color: i for i, color in enumerate(unique_colors)}
            labels = [color_to_label[color] for color in enhanced_colors]
            return labels, unique_colors
        
        # 转换为numpy数组
        colors_array = np.array(enhanced_colors, dtype=np.float32)
        
        if weights is None:
            weights = np.ones(len(element_colors))
        else:
            weights = np.array(weights)
        
        # 使用加权K-means聚类
        labels, centroids = self._weighted_kmeans_gpu(colors_array, k, weights)
        
        # 转换回整数RGB，确保是整数类型
        centroids = [(int(round(c[0])), int(round(c[1])), int(round(c[2]))) for c in centroids]
        
        return labels, centroids
    
    def _enhance_saturation(self, colors: List[Tuple[int, int, int]], factor: float) -> List[Tuple[int, int, int]]:
        """增强颜色饱和度"""
        enhanced = []
        
        for r, g, b in colors:
            # 转换到HSV
            rgb = np.array([[[r, g, b]]], dtype=np.uint8)
            hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
            
            # 增强饱和度
            hsv[0, 0, 1] = np.clip(hsv[0, 0, 1] * factor, 0, 255)
            
            # 转换回RGB
            hsv = hsv.astype(np.uint8)
            rgb_enhanced = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
            
            # 确保返回整数元组
            enhanced.append((int(rgb_enhanced[0, 0, 0]), 
                           int(rgb_enhanced[0, 0, 1]), 
                           int(rgb_enhanced[0, 0, 2])))
        
        return enhanced
    
    def _weighted_kmeans_gpu(self, colors: np.ndarray, k: int, weights: np.ndarray, 
                            max_iters: int = 100, tol: float = 1e-4) -> Tuple[np.ndarray, np.ndarray]:
        """
        GPU加速的加权K-means聚类
        """
        n_samples = colors.shape[0]
        
        # 转换到GPU
        colors_gpu = torch.from_numpy(colors).to(self.device)
        weights_gpu = torch.from_numpy(weights).float().to(self.device)
        
        # 初始化聚类中心（使用k-means++策略）
        centroids_gpu = self._kmeans_plusplus_init(colors_gpu, k, weights_gpu)
        
        labels = torch.zeros(n_samples, dtype=torch.long, device=self.device)
        
        for iteration in range(max_iters):
            # E步：分配样本到最近的聚类中心
            distances = torch.cdist(colors_gpu, centroids_gpu)
            new_labels = torch.argmin(distances, dim=1)
            
            # 检查收敛
            if torch.all(labels == new_labels):
                break
            
            labels = new_labels
            
            # M步：更新聚类中心（加权平均）
            new_centroids = torch.zeros_like(centroids_gpu)
            for i in range(k):
                mask = (labels == i)
                if torch.any(mask):
                    cluster_colors = colors_gpu[mask]
                    cluster_weights = weights_gpu[mask]
                    # 加权平均
                    weighted_sum = torch.sum(cluster_colors * cluster_weights.unsqueeze(1), dim=0)
                    weight_sum = torch.sum(cluster_weights)
                    new_centroids[i] = weighted_sum / weight_sum
                else:
                    # 如果某个聚类为空，保持原中心
                    new_centroids[i] = centroids_gpu[i]
            
            # 检查中心点变化
            center_shift = torch.max(torch.norm(new_centroids - centroids_gpu, dim=1))
            centroids_gpu = new_centroids
            
            if center_shift < tol:
                break
        
        # 转换回CPU
        labels_cpu = labels.cpu().numpy()
        centroids_cpu = centroids_gpu.cpu().numpy()
        
        return labels_cpu, centroids_cpu
    
    def _kmeans_plusplus_init(self, colors: torch.Tensor, k: int, weights: torch.Tensor) -> torch.Tensor:
        """K-means++初始化策略"""
        n_samples = colors.shape[0]
        centroids = torch.zeros((k, 3), device=self.device)
        
        # 根据权重选择第一个中心
        probs = weights / weights.sum()
        first_idx = torch.multinomial(probs, 1).item()
        centroids[0] = colors[first_idx]
        
        for i in range(1, k):
            # 计算每个点到最近中心的距离
            distances = torch.cdist(colors, centroids[:i])
            min_distances, _ = torch.min(distances, dim=1)
            
            # 根据距离和权重计算概率
            probs = (min_distances ** 2) * weights
            probs = probs / probs.sum()
            
            # 选择下一个中心
            next_idx = torch.multinomial(probs, 1).item()
            centroids[i] = colors[next_idx]
        
        return centroids
    
    def quantize_with_protection(self,
                                element_colors: List[Tuple[int, int, int]],
                                k: int,
                                protected_indices: List[int],
                                protection_weight: float = 5.0,
                                saturation_factor: float = 1.2) -> Tuple[List[int], List[Tuple[int, int, int]]]:
        """
        带保护机制的颜色量化（用于保护眼睛、嘴唇等重要特征）
        
        参数:
            element_colors: 元素颜色列表
            k: 目标颜色数
            protected_indices: 需要保护的元素索引
            protection_weight: 保护权重
            saturation_factor: 饱和度增强系数
        """
        # 创建权重数组
        weights = np.ones(len(element_colors))
        for idx in protected_indices:
            if idx < len(weights):
                weights[idx] = protection_weight
        
        return self.quantize_colors(element_colors, k, weights.tolist(), saturation_factor)

class RegionAwareSVGQuantizer:
    """区域感知的SVG颜色量化器"""
    
    def __init__(self, use_gpu: bool = True):
        self.quantizer = SVGColorQuantizer(use_gpu)
        
    def quantize_by_regions(self,
                           skin_elements: List[Tuple[int, Tuple[int, int, int]]],
                           env_elements: List[Tuple[int, Tuple[int, int, int]]],
                           k_skin: int,
                           k_env: int,
                           importance_weights: Dict[int, float],
                           saturation_factor: float = 1.2) -> Dict[int, Tuple[int, int, int]]:
        """
        分区域进行颜色量化
        
        参数:
            skin_elements: [(元素索引, 颜色), ...]
            env_elements: [(元素索引, 颜色), ...]
            k_skin: 皮肤区域目标颜色数
            k_env: 环境区域目标颜色数
            importance_weights: 元素重要性权重
            saturation_factor: 饱和度增强系数
            
        返回:
            元素索引到新颜色的映射
        """
        color_mapping = {}
        
        # 处理皮肤区域
        if skin_elements:
            skin_indices = [idx for idx, _ in skin_elements]
            skin_colors = [color for _, color in skin_elements]
            skin_weights = [importance_weights.get(idx, 1.0) for idx in skin_indices]
            
            # 识别可能的面部特征
            protected_indices = []
            for i, (idx, color) in enumerate(skin_elements):
                if importance_weights.get(idx, 1.0) > 2.0:  # 高权重元素需要保护
                    protected_indices.append(i)
            
            labels, centroids = self.quantizer.quantize_with_protection(
                skin_colors, k_skin, protected_indices, 
                protection_weight=5.0, saturation_factor=saturation_factor
            )
            
            for i, idx in enumerate(skin_indices):
                color_mapping[idx] = centroids[labels[i]]
        
        # 处理环境区域
        if env_elements:
            env_indices = [idx for idx, _ in env_elements]
            env_colors = [color for _, color in env_elements]
            env_weights = [importance_weights.get(idx, 1.0) for idx in env_indices]
            
            labels, centroids = self.quantizer.quantize_colors(
                env_colors, k_env, env_weights, saturation_factor=saturation_factor
            )
            
            for i, idx in enumerate(env_indices):
                color_mapping[idx] = centroids[labels[i]]
        
        return color_mapping