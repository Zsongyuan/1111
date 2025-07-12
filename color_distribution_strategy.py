# color_distribution_strategy.py
"""
高级颜色分配策略模块，基于色彩复杂度进行动态分配。
"""
import numpy as np
from typing import Tuple, List
import cv2

# 【新增】
def _calculate_color_diversity(elements: List[Tuple[int, Tuple[int, int, int]]]) -> float:
    """计算一个区域内颜色的标准差作为色彩复杂度的度量"""
    if len(elements) < 2:
        return 1.0  # 如果颜色太少，则复杂度为1

    colors = [e[1] for e in elements]
    # 在LAB空间计算标准差更符合视觉感知
    lab_colors = [cv2.cvtColor(np.uint8([[list(c)]]), cv2.COLOR_RGB2LAB)[0][0] for c in colors]
    std_dev = np.std(lab_colors, axis=0)
    # 使用L, a, b三个通道标准差的平均值作为最终复杂度得分
    complexity = np.mean(std_dev)
    
    # 将复杂度归一化到一个合理的范围（例如1到5），避免极端值
    return np.clip(complexity / 10.0, 1.0, 5.0)

class ColorDistributionStrategy:
    def __init__(self, 
                 skin_ratio_threshold: float = 0.1,
                 min_skin_colors: int = 5,
                 default_skin_weight: int = 4, # 降低基础权重
                 enhanced_skin_weight: int = 6,
                 min_colors_per_region: int = 3):
        self.skin_ratio_threshold = skin_ratio_threshold
        self.min_skin_colors = min_skin_colors
        self.default_skin_weight = default_skin_weight
        self.enhanced_skin_weight = enhanced_skin_weight
        self.min_colors_per_region = min_colors_per_region
        
    def calculate_distribution(self,
                             target_k: int,
                             skin_ratio: float,
                             facial_elements: List,
                             env_elements: List) -> Tuple[int, int]:
        n_facial = len(facial_elements)
        n_env = len(env_elements)
        
        if n_facial == 0: return 0, target_k
        if n_env == 0: return target_k, 0

        # 【核心重构】计算色彩复杂度
        facial_diversity = _calculate_color_diversity(facial_elements)
        env_diversity = _calculate_color_diversity(env_elements)
        print(f"色彩复杂度: 面部={facial_diversity:.2f}, 环境={env_diversity:.2f}")

        # 计算元素数量权重
        facial_elem_weight = self.enhanced_skin_weight if skin_ratio > self.skin_ratio_threshold else self.default_skin_weight
        
        # 【核心重构】最终权重 = 元素权重 * 复杂度权重
        weighted_facial = n_facial * facial_elem_weight * facial_diversity
        weighted_env = n_env * 1.0 * env_diversity # 环境的基础权重为1
        
        total_weight = weighted_facial + weighted_env
        if total_weight == 0: return target_k // 2, target_k - (target_k // 2)

        k_facial = int(round(target_k * weighted_facial / total_weight))
        k_env = target_k - k_facial
        
        print(f"初始分配: 面部={k_facial}, 环境={k_env}")
        return self._apply_constraints(k_facial, k_env, target_k)
    
    def _apply_constraints(self, k_facial: int, k_env: int, target_k: int) -> Tuple[int, int]:
        min_k = self.min_colors_per_region
        
        if target_k < min_k * 2:
            k_facial = max(1, k_facial); k_env = target_k - k_facial
            print(f"总颜色数不足，跳过最小约束 -> 面部: {k_facial}, 环境: {k_env}")
            return k_facial, k_env

        if k_facial < min_k:
            transfer = min(min_k - k_facial, k_env - min_k)
            if transfer > 0: k_facial += transfer; k_env -= transfer
        
        if k_env < min_k:
            transfer = min(min_k - k_env, k_facial - min_k)
            if transfer > 0: k_env += transfer; k_facial -= transfer
        
        k_facial = target_k - k_env
        k_facial = max(1, k_facial); k_env = max(1, k_env)

        print(f"约束应用后 -> 面部: {k_facial}, 环境: {k_env}")
        return k_facial, k_env
    
    def suggest_adjustment(self, actual, target, current) -> int:
        if abs(actual - target) <= 1: return current
        return max(1, current + (target - actual))