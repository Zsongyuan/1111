# test_robust_palette.py
"""
测试稳健色板选择算法
"""
import sys
import numpy as np
from typing import List, Tuple

def test_palette_selection_robustness():
    """测试色板选择的稳健性和多样性"""
    print("="*60)
    print("测试稳健色板选择算法")
    print("="*60)
    
    try:
        from svg_palette_matching import RobustSVGPaletteMatcher
        
        matcher = RobustSVGPaletteMatcher()
        
        # 设计多种不同特征的测试用例
        test_cases = [
            ("浅色冷调皮肤", [
                (255, 230, 205), (250, 225, 200), (245, 220, 195),
                (240, 215, 190), (235, 210, 185)
            ]),
            ("中等暖调皮肤", [
                (210, 180, 140), (205, 175, 135), (200, 170, 130),
                (195, 165, 125), (190, 160, 120)
            ]),
            ("深色皮肤", [
                (160, 120, 90), (155, 115, 85), (150, 110, 80),
                (145, 105, 75), (140, 100, 70)
            ]),
            ("红润皮肤", [
                (230, 170, 140), (225, 165, 135), (220, 160, 130),
                (215, 155, 125), (210, 150, 120)
            ]),
            ("偏黄皮肤", [
                (230, 200, 150), (225, 195, 145), (220, 190, 140),
                (215, 185, 135), (210, 180, 130)
            ]),
            ("苍白皮肤", [
                (245, 235, 225), (240, 230, 220), (235, 225, 215),
                (230, 220, 210), (225, 215, 205)
            ]),
            ("橄榄色皮肤", [
                (180, 160, 120), (175, 155, 115), (170, 150, 110),
                (165, 145, 105), (160, 140, 100)
            ]),
            ("粉调皮肤", [
                (220, 180, 170), (215, 175, 165), (210, 170, 160),
                (205, 165, 155), (200, 160, 150)
            ])
        ]
        
        selected_palettes = []
        detailed_results = []
        
        for test_name, test_colors in test_cases:
            print(f"\n{'='*40}")
            print(f"测试用例: {test_name}")
            print(f"输入颜色: {test_colors[:2]}... (共{len(test_colors)}色)")
            
            # 选择色板
            selected = matcher._select_best_skin_palette_robust(test_colors)
            selected_palettes.append(selected)
            detailed_results.append((test_name, selected, test_colors))
            
            print(f"选择结果: {selected}")
        
        # 分析结果
        unique_palettes = len(set(selected_palettes))
        total_cases = len(test_cases)
        
        print(f"\n{'='*60}")
        print("结果分析")
        print(f"{'='*60}")
        print(f"测试用例总数: {total_cases}")
        print(f"选择的不同色板数: {unique_palettes}")
        print(f"多样性得分: {unique_palettes / total_cases * 100:.1f}%")
        
        # 详细结果展示
        print(f"\n详细选择结果:")
        palette_count = {}
        for test_name, selected, _ in detailed_results:
            print(f"  {test_name:15} -> {selected}")
            palette_count[selected] = palette_count.get(selected, 0) + 1
        
        print(f"\n色板使用频率:")
        for palette, count in sorted(palette_count.items()):
            print(f"  {palette}: {count} 次 ({count/total_cases*100:.1f}%)")
        
        # 评估标准
        success_criteria = {
            "多样性": unique_palettes >= 5,  # 至少选择5个不同色板
            "无偏向": max(palette_count.values()) <= total_cases * 0.4,  # 没有色板被选择超过40%
            "合理性": all(selected != "SKIN-A" or test_name == "浅色冷调皮肤" 
                         for test_name, selected, _ in detailed_results)  # SKIN-A只应该用于特定情况
        }
        
        print(f"\n评估结果:")
        all_passed = True
        for criterion, passed in success_criteria.items():
            status = "✓ 通过" if passed else "✗ 失败"
            print(f"  {criterion}: {status}")
            if not passed:
                all_passed = False
        
        return all_passed
        
    except Exception as e:
        print(f"✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_matching_methods():
    """测试各种匹配方法的效果"""
    print(f"\n{'='*60}")
    print("测试匹配方法的有效性")
    print(f"{'='*60}")
    
    try:
        from svg_palette_matching import RobustSVGPaletteMatcher
        
        matcher = RobustSVGPaletteMatcher()
        
        # 测试颜色
        test_colors = [(210, 180, 140), (200, 170, 130), (195, 165, 125)]
        
        print(f"测试颜色: {test_colors}")
        
        # 转换为LAB
        input_lab = [matcher._rgb_to_lab(c) for c in test_colors]
        
        # 测试方法1: 匹配质量
        print(f"\n方法1: 匹配质量评分")
        scores1 = matcher._evaluate_by_matching_quality(input_lab)
        top3_method1 = sorted(scores1.items(), key=lambda x: x[1], reverse=True)[:3]
        for palette, score in top3_method1:
            print(f"  {palette}: {score:.3f}")
        
        # 测试方法2: 覆盖度
        print(f"\n方法2: 覆盖度评分")
        scores2 = matcher._evaluate_by_coverage_quality(test_colors)
        top3_method2 = sorted(scores2.items(), key=lambda x: x[1], reverse=True)[:3]
        for palette, score in top3_method2:
            print(f"  {palette}: {score:.3f}")
        
        # 测试方法3: 色温
        print(f"\n方法3: 色温匹配评分")
        scores3 = matcher._evaluate_by_color_temperature(test_colors)
        top3_method3 = sorted(scores3.items(), key=lambda x: x[1], reverse=True)[:3]
        for palette, score in top3_method3:
            print(f"  {palette}: {score:.3f}")
        
        # 检查各方法是否给出了不同的结果
        method_diversity = len(set([top3_method1[0][0], top3_method2[0][0], top3_method3[0][0]]))
        print(f"\n方法多样性: {method_diversity} 种不同的首选色板")
        
        return method_diversity >= 2  # 至少有2种方法给出不同结果
        
    except Exception as e:
        print(f"✗ 方法测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_edge_cases():
    """测试边界情况"""
    print(f"\n{'='*60}")
    print("测试边界情况")
    print(f"{'='*60}")
    
    try:
        from svg_palette_matching import RobustSVGPaletteMatcher
        
        matcher = RobustSVGPaletteMatcher()
        
        edge_cases = [
            ("空颜色列表", []),
            ("单一颜色", [(200, 150, 120)]),
            ("相同颜色", [(200, 150, 120), (200, 150, 120), (200, 150, 120)]),
            ("极端颜色", [(0, 0, 0), (255, 255, 255)]),
            ("大量颜色", [(i*5, i*4, i*3) for i in range(50)])
        ]
        
        all_passed = True
        
        for test_name, test_colors in edge_cases:
            print(f"\n测试: {test_name}")
            try:
                selected = matcher._select_best_skin_palette_robust(test_colors)
                print(f"  结果: {selected}")
                
                # 验证选择的色板确实存在
                if selected not in matcher.skin_palettes:
                    print(f"  ✗ 选择了不存在的色板: {selected}")
                    all_passed = False
                else:
                    print(f"  ✓ 选择了有效色板")
                    
            except Exception as e:
                print(f"  ✗ 处理失败: {e}")
                all_passed = False
        
        return all_passed
        
    except Exception as e:
        print(f"✗ 边界测试失败: {e}")
        return False

def main():
    """运行所有测试"""
    print("稳健色板选择算法测试")
    print("时间:", __import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    
    tests = [
        ("色板选择多样性", test_palette_selection_robustness),
        ("匹配方法有效性", test_matching_methods),
        ("边界情况处理", test_edge_cases),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"✗ {test_name} 执行失败: {e}")
            results.append((test_name, False))
    
    # 汇总结果
    print(f"\n{'='*60}")
    print("测试结果汇总")
    print(f"{'='*60}")
    
    passed = 0
    for test_name, success in results:
        status = "✓ 通过" if success else "✗ 失败"
        print(f"{test_name}: {status}")
        if success:
            passed += 1
    
    success_rate = passed / len(results) * 100
    print(f"\n总体成功率: {passed}/{len(results)} ({success_rate:.1f}%)")
    
    if success_rate >= 80:
        print(f"\n🎉 稳健色板选择算法测试通过！")
        print(f"\n主要改进:")
        print(f"  • 基于实际颜色匹配质量评分")
        print(f"  • 多方法综合评估（匹配质量+覆盖度+色温）")
        print(f"  • 智能回退机制")
        print(f"  • 处理各种边界情况")
        print(f"\n现在可以:")
        print(f"  python validate_fixes.py  # 验证完整系统")
        print(f"  python svg_main.py        # 处理SVG文件")
    else:
        print(f"\n❌ 测试失败，需要进一步优化算法")
    
    return success_rate >= 80

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)