# svg_color_quantizer.py
"""
SVG颜色量化模块：使用GPU加速对SVG元素的颜色进行聚类和量化
"""
import numpy as np
import torch
from typing import List, Tuple, Dict, Optional
import cv2

class SVGColorQuantizer:
    """基于GPU的SVG颜色量化器"""
    
    def __init__(self, use_gpu: bool = True):
        self.device = torch.device('cuda' if use_gpu and torch.cuda.is_available() else 'cpu')
        print(f"颜色量化器使用设备: {self.device}")
        
    def quantize_colors(self, 
                       element_colors: List[Tuple[int, int, int]], 
                       k: int,
                       weights: Optional[List[float]] = None,
                       saturation_factor: float = 1.2) -> Tuple[List[int], List[Tuple[int, int, int]]]:
        if not element_colors or k <= 0:
            return [], []
        
        enhanced_colors = self._enhance_saturation(element_colors, saturation_factor)
        unique_colors = list(set(map(tuple, enhanced_colors))) # 确保唯一性
        
        if len(unique_colors) <= k:
            color_to_label = {color: i for i, color in enumerate(unique_colors)}
            labels = [color_to_label[tuple(color)] for color in enhanced_colors]
            # 【修复】确保此处返回的是元组列表，而不是列表的列表
            return labels, unique_colors
        
        colors_array = np.array(enhanced_colors, dtype=np.float32)
        weights_array = np.array(weights if weights is not None else np.ones(len(element_colors)))
        
        labels, centroids = self._weighted_kmeans_gpu(colors_array, k, weights_array)
        centroids_int = [(int(round(c[0])), int(round(c[1])), int(round(c[2]))) for c in centroids]
        return labels, centroids_int

    # ... a​​​​​​​ll other methods in this file remain unchanged ...
    def _enhance_saturation(self, colors, factor):
        enhanced = []
        for r, g, b in colors:
            rgb = np.uint8([[[r, g, b]]]); hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
            hsv[0,0,1] = np.clip(hsv[0,0,1]*factor, 0, 255)
            rgb_enhanced = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
            enhanced.append(tuple(rgb_enhanced[0,0]))
        return enhanced

    def _weighted_kmeans_gpu(self, colors: np.ndarray, k: int, weights: np.ndarray, 
                            max_iters: int = 100, tol: float = 1e-4) -> Tuple[np.ndarray, np.ndarray]:
        colors_gpu = torch.from_numpy(colors).to(self.device)
        weights_gpu = torch.from_numpy(weights).float().to(self.device)
        centroids_gpu = self._kmeans_plusplus_init(colors_gpu, k, weights_gpu)
        for _ in range(max_iters):
            distances = torch.cdist(colors_gpu, centroids_gpu); prev_labels = labels if 'labels' in locals() else None
            labels = torch.argmin(distances, dim=1)
            if prev_labels is not None and torch.all(labels == prev_labels): break
            new_centroids = torch.zeros_like(centroids_gpu)
            for i in range(k):
                mask = (labels == i)
                if torch.any(mask):
                    new_centroids[i] = torch.sum(colors_gpu[mask]*weights_gpu[mask].unsqueeze(1), dim=0) / torch.sum(weights_gpu[mask])
                else: new_centroids[i] = centroids_gpu[i] 
            if torch.max(torch.norm(new_centroids - centroids_gpu, dim=1)) < tol: break
            centroids_gpu = new_centroids
        return labels.cpu().numpy(), centroids_gpu.cpu().numpy()

    def _kmeans_plusplus_init(self, colors: torch.Tensor, k: int, weights: torch.Tensor) -> torch.Tensor:
        n_samples = colors.shape[0]; centroids = torch.zeros((k, 3), device=self.device)
        probs = weights/weights.sum(); first_idx = torch.multinomial(probs, 1).item()
        centroids[0] = colors[first_idx]
        for i in range(1, k):
            min_distances, _ = torch.min(torch.cdist(colors, centroids[:i]), dim=1)
            probs = (min_distances**2) * weights; probs /= probs.sum()
            centroids[i] = colors[torch.multinomial(probs, 1).item()]
        return centroids

class RegionAwareSVGQuantizer:
    def __init__(self, use_gpu: bool = True):
        self.quantizer = SVGColorQuantizer(use_gpu)
        
    def quantize_by_regions(self,
                           facial_elements: List[Tuple[int, Tuple[int, int, int]]],
                           env_elements: List[Tuple[int, Tuple[int, int, int]]],
                           k_facial: int,
                           k_env: int,
                           importance_weights: Dict[int, float],
                           saturation_factor: float = 1.2) -> Dict[int, Tuple[int, int, int]]:
        color_mapping = {}

        if facial_elements and k_facial > 0:
            facial_indices = [idx for idx, _ in facial_elements]
            facial_colors = [color for _, color in facial_elements]
            facial_weights = [importance_weights.get(idx, 1.0) for idx in facial_indices]
            
            labels, centroids = self.quantizer.quantize_colors(
                facial_colors, k_facial, facial_weights, saturation_factor)
            
            if labels is not None and len(labels) > 0:
                for i, idx in enumerate(facial_indices):
                    color_mapping[idx] = centroids[labels[i]]

        if env_elements and k_env > 0:
            env_indices = [idx for idx, _ in env_elements]
            env_colors = [color for _, color in env_elements]
            env_weights = [importance_weights.get(idx, 1.0) for idx in env_indices]
            
            labels, centroids = self.quantizer.quantize_colors(
                env_colors, k_env, env_weights, saturation_factor)

            if labels is not None and len(labels) > 0:
                for i, idx in enumerate(env_indices):
                    color_mapping[idx] = centroids[labels[i]]
        
        return color_mapping