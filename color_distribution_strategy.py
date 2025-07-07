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
                 min_skin_colors: int = 8,      # 也可以适当提高最小颜色数
                 default_skin_weight: int = 12,  # <--- 将默认权重从3提高到12
                 enhanced_skin_weight: int = 16): # <--- 也可以一并提高
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
        计算颜色分配 (已增加对无皮肤情况的处理)
        """
        # --- 核心修复：增加对无皮肤情况的特殊处理 ---
        # 如果一开始就没有检测到任何皮肤元素，则直接将所有颜色分配给环境
        if n_skin_elements == 0:
            print("未检测到皮肤元素，所有颜色将分配给环境区域。")
            k_skin = 0
            k_env = target_k
            return k_skin, k_env
        # --- 修复结束 ---
            
        # (如果检测到皮肤元素，则继续执行原有的权重分配逻辑)
        if skin_ratio > self.skin_ratio_threshold:
            weight_skin = self.enhanced_skin_weight
            print(f"皮肤区域占比 {skin_ratio:.1%} > {self.skin_ratio_threshold:.1%}, 使用增强权重")
        else:
            weight_skin = self.default_skin_weight
            print(f"皮肤区域占比 {skin_ratio:.1%} <= {self.skin_ratio_threshold:.1%}, 使用默认权重")
        
        if skin_pixels > 0 and env_pixels > 0:
            weighted_skin = skin_pixels * weight_skin
            weighted_env = env_pixels
            print(f"基于像素计算: 皮肤像素={skin_pixels}, 环境像素={env_pixels}")
        else:
            weighted_skin = n_skin_elements * weight_skin
            weighted_env = n_env_elements
            print(f"基于元素计算: 皮肤元素={n_skin_elements}, 环境元素={n_env_elements}")
        
        total_weight = weighted_skin + weighted_env
        
        if total_weight == 0:
            # 当所有元素数量都为0时（虽然不太可能发生），进行均分
            k_skin = target_k // 2
        else:
            k_skin = int(round(target_k * weighted_skin / total_weight))
        
        k_env = target_k - k_skin
        print(f"初始分配: 皮肤={k_skin}, 环境={k_env}")
        
        k_skin, k_env = self._apply_constraints(k_skin, k_env, target_k, skin_ratio)
        
        return k_skin, k_env
    
    def _apply_constraints(self, k_skin: int, k_env: int, target_k: int, skin_ratio: float) -> Tuple[int, int]:
        """(已修复) 应用分配约束，确保最小颜色数规则总是被正确应用"""
        
        # 1. 检查是否需要应用最小颜色数规则
        # 只有在计算出的皮肤颜色数低于我们设定的下限时，才进行干预
        if k_skin < self.min_skin_colors:
            print(f"触发最小颜色数规则：计算出的皮肤颜色({k_skin})少于设定的最小值({self.min_skin_colors})。")
            # 将皮肤颜色数提升到设定的最小值，但不能超过总数减一（至少给环境留一个）
            k_skin = min(self.min_skin_colors, target_k - 1)
            print(f"  > 调整后：皮肤颜色数提升至 {k_skin}")

        # 2. 确保环境区域至少有1种颜色
        # 即使上一步提升了皮肤颜色数，也要保证环境的最低配额
        if k_skin >= target_k:
            k_skin = target_k - 1

        # 3. 基于可能已更新的k_skin，重新计算k_env，以确保总数不变
        k_env = target_k - k_skin

        # 4. 对k_env也设置一个最低值1的保护
        if k_env < 1:
            k_env = 1
            # 如果因为提升k_env导致总数超了，就从k_skin里减回来
            k_skin = target_k - k_env

        print(f"约束应用后 -> 皮肤: {k_skin}, 环境: {k_env}, 总计: {k_skin + k_env}")
        
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