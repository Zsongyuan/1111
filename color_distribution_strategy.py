# color_distribution_strategy.py
"""
高级颜色分配策略模块
基于位图版本的算法改进
"""
import numpy as np
from typing import Tuple, Dict, List

class ColorDistributionStrategy:
    """颜色分配策略类"""
    
    def __init__(self, 
                 skin_ratio_threshold: float = 0.1,
                 min_skin_colors: int = 5,
                 default_skin_weight: int = 3,
                 enhanced_skin_weight: int = 5):
        self.skin_ratio_threshold = skin_ratio_threshold
        self.min_skin_colors = min_skin_colors
        self.default_skin_weight = default_skin_weight
        self.enhanced_skin_weight = enhanced_skin_weight
        
    def calculate_distribution(self,
                             target_k: int,
                             skin_ratio: float,
                             n_skin_elements: int,
                             n_env_elements: int,
                             skin_pixels: int = 0,
                             env_pixels: int = 0) -> Tuple[int, int]:
        """
        计算颜色分配
        
        参数:
            target_k: 目标总颜色数
            skin_ratio: 皮肤区域占比（0-1）
            n_skin_elements: 皮肤元素数量
            n_env_elements: 环境元素数量
            skin_pixels: 皮肤像素数（可选，用于更精确的计算）
            env_pixels: 环境像素数（可选）
        """
        # 选择权重策略
        if skin_ratio > self.skin_ratio_threshold:
            # 皮肤是主体，使用增强权重
            weight_skin = self.enhanced_skin_weight
            print(f"皮肤区域占比 {skin_ratio:.1%} > {self.skin_ratio_threshold:.1%}, 使用增强权重")
        else:
            # 皮肤是次要部分，使用默认权重
            weight_skin = self.default_skin_weight
            print(f"皮肤区域占比 {skin_ratio:.1%} <= {self.skin_ratio_threshold:.1%}, 使用默认权重")
        
        # 计算加权值
        # 如果提供了像素数，优先使用像素数计算
        if skin_pixels > 0 and env_pixels > 0:
            weighted_skin = skin_pixels * weight_skin
            weighted_env = env_pixels
            print(f"基于像素计算: 皮肤像素={skin_pixels}, 环境像素={env_pixels}")
        else:
            # 否则使用元素数量
            weighted_skin = n_skin_elements * weight_skin
            weighted_env = n_env_elements
            print(f"基于元素计算: 皮肤元素={n_skin_elements}, 环境元素={n_env_elements}")
        
        total_weight = weighted_skin + weighted_env
        
        # 计算初始分配
        if total_weight == 0:
            k_skin = target_k // 2
            k_env = target_k - k_skin
        else:
            # 基于权重比例分配
            k_skin = int(round(target_k * weighted_skin / total_weight))
            k_env = target_k - k_skin
        
        print(f"初始分配: 皮肤={k_skin}, 环境={k_env}")
        
        # 应用约束条件
        k_skin, k_env = self._apply_constraints(k_skin, k_env, target_k, skin_ratio)
        
        return k_skin, k_env
    
    def _apply_constraints(self, k_skin: int, k_env: int, target_k: int, skin_ratio: float) -> Tuple[int, int]:
        """应用分配约束"""
        # 确保至少各有1种颜色
        k_skin = max(1, k_skin)
        k_env = max(1, k_env)
        
        # 当皮肤是主体时，确保有足够的颜色
        if skin_ratio > self.skin_ratio_threshold and k_skin < self.min_skin_colors:
            k_skin = min(self.min_skin_colors, target_k - 1)
            k_env = target_k - k_skin
            print(f"皮肤是主体，调整为最少 {self.min_skin_colors} 种颜色")
        
        # 再次确保约束
        if k_env < 1:
            k_env = 1
            k_skin = target_k - k_env
        
        # 确保总数正确
        if k_skin + k_env != target_k:
            # 调整环境颜色数以匹配总数
            k_env = target_k - k_skin
        
        # 最终检查
        k_skin = max(1, min(k_skin, target_k - 1))
        k_env = max(1, target_k - k_skin)
        
        print(f"最终分配: 皮肤={k_skin}, 环境={k_env}, 总计={k_skin + k_env}")
        
        return k_skin, k_env
    
    def suggest_adjustment(self, 
                          actual_colors: int, 
                          target_colors: int,
                          current_k: int) -> int:
        """
        建议下一次迭代的颜色数调整
        
        参数:
            actual_colors: 实际得到的颜色数
            target_colors: 目标颜色数
            current_k: 当前使用的k值
        """
        diff = target_colors - actual_colors
        
        if abs(diff) <= 2:
            # 已经足够接近
            return current_k
        
        if actual_colors < target_colors:
            # 颜色太少，需要增加
            # 但不要增加太多，避免过度调整
            adjustment = min(diff, max(2, diff // 2))
            new_k = current_k + adjustment
            print(f"颜色不足 ({actual_colors} < {target_colors})，建议增加 {adjustment}")
        else:
            # 颜色太多，需要减少
            # 同样避免过度调整
            adjustment = min(diff, max(2, diff // 2))
            new_k = max(target_colors // 2, current_k - adjustment)
            print(f"颜色过多 ({actual_colors} > {target_colors})，建议减少 {adjustment}")
        
        return new_k

# 测试代码
if __name__ == '__main__':
    strategy = ColorDistributionStrategy()
    
    # 测试场景1：皮肤占比低
    print("=" * 60)
    print("测试场景1：皮肤占比低")
    k_skin, k_env = strategy.calculate_distribution(
        target_k=48,
        skin_ratio=0.05,
        n_skin_elements=100,
        n_env_elements=900
    )
    
    # 测试场景2：皮肤占比高
    print("\n" + "=" * 60)
    print("测试场景2：皮肤占比高")
    k_skin, k_env = strategy.calculate_distribution(
        target_k=48,
        skin_ratio=0.25,
        n_skin_elements=400,
        n_env_elements=600
    )
    
    # 测试场景3：极端情况
    print("\n" + "=" * 60)
    print("测试场景3：极端情况（很少的总颜色）")
    k_skin, k_env = strategy.calculate_distribution(
        target_k=6,
        skin_ratio=0.15,
        n_skin_elements=200,
        n_env_elements=800
    )