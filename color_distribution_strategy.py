# color_distribution_strategy.py
"""
(已重构) “面部优先”颜色分配策略。
"""
import numpy as np
from typing import Tuple, List
import cv2

class FacialPriorityColorDistribution:
    def __init__(self, 
                 simple_env_k: int = 8,
                 complex_env_k: int = 15,
                 variance_threshold: float = 50.0):
        self.simple_env_k = simple_env_k
        self.complex_env_k = complex_env_k
        self.variance_threshold = variance_threshold
        
    def _calculate_color_variance(self, colors: List[Tuple[int, int, int]]) -> float:
        """计算颜色列表的方差，作为复杂度的衡量标准"""
        if len(colors) < 2:
            return 0.0
        # 在LAB空间计算方差更符合视觉感知
        lab_colors = [cv2.cvtColor(np.uint8([[list(c)]]), cv2.COLOR_RGB2LAB)[0][0] for c in colors]
        return np.mean(np.var(lab_colors, axis=0))

    def analyze_and_allocate(self,
                             target_k: int,
                             env_colors: List[Tuple[int, int, int]]) -> Tuple[int, int]:
        """
        两阶段分配：先评估环境，再将剩余预算全部分配给面部。
        """
        # --- 第一阶段：环境评估与分配 ---
        env_variance = self._calculate_color_variance(env_colors)
        
        if env_variance < self.variance_threshold:
            # 环境色彩单一，分配较少的颜色
            k_env = min(self.simple_env_k, len(set(env_colors)) if env_colors else 1)
            print(f"环境色彩单一 (方差: {env_variance:.2f})，分配 {k_env} 种颜色。")
        else:
            # 环境色彩复杂，分配较多的颜色
            k_env = min(self.complex_env_k, len(set(env_colors)) if env_colors else 1)
            print(f"环境色彩复杂 (方差: {env_variance:.2f})，分配 {k_env} 种颜色。")

        # --- 第二阶段：面部优先分配 ---
        # 确保k_env不超过总数的一半，为面部保留充足预算
        k_env = min(k_env, target_k // 2)
        
        k_facial = target_k - k_env
        print(f"面部优先：将剩余 {k_facial} 种颜色全部分配给面部区域。")

        return k_facial, k_env