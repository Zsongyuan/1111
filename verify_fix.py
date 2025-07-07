# verify_fix.py
"""
验证颜色类型修复是否有效
"""
import os
import sys

def verify_imports():
    """验证所有模块可以正常导入"""
    print("验证模块导入...")
    modules = [
        'svg_parser',
        'svg_region_mapper', 
        'svg_color_quantizer',
        'svg_palette_matching',
        'svg_output',
        'svg_main',
        'color_distribution_strategy'
    ]
    
    for module in modules:
        try:
            __import__(module)
            print(f"✓ {module}")
        except ImportError as e:
            print(f"✗ {module}: {e}")
            return False
        except Exception as e:
            print(f"✗ {module}: {e}")
            return False
    
    return True

def test_color_conversion():
    """测试颜色转换"""
    print("\n测试颜色转换...")
    
    from svg_output import SVGOutput
    output = SVGOutput(None)
    
    # 测试各种输入类型
    test_inputs = [
        (255, 128, 0),        # 整数
        (255.0, 128.0, 0.0),  # 浮点数
        (255.5, 128.3, 0.9),  # 带小数的浮点数
    ]
    
    for rgb in test_inputs:
        try:
            hex_color = output._rgb_to_hex(rgb)
            print(f"✓ {rgb} -> {hex_color}")
        except Exception as e:
            print(f"✗ {rgb} -> 错误: {e}")
            return False
    
    return True

def test_color_strategy():
    """测试颜色分配策略"""
    print("\n测试颜色分配策略...")
    
    try:
        from color_distribution_strategy import ColorDistributionStrategy
        strategy = ColorDistributionStrategy()
        
        # 测试基本分配
        k_skin, k_env = strategy.calculate_distribution(
            target_k=48,
            skin_ratio=0.087,  # 8.7% 如错误日志中显示
            n_skin_elements=345,
            n_env_elements=7920
        )
        
        total = k_skin + k_env
        print(f"✓ 分配结果: 皮肤={k_skin}, 环境={k_env}, 总计={total}")
        
        if total != 48:
            print(f"✗ 总颜色数不正确: {total} != 48")
            return False
            
    except Exception as e:
        print(f"✗ 颜色分配策略错误: {e}")
        return False
    
    return True

def main():
    """运行所有验证"""
    print("="*60)
    print("颜色类型修复验证")
    print("="*60)
    
    tests = [
        ("模块导入", verify_imports),
        ("颜色转换", test_color_conversion),
        ("颜色策略", test_color_strategy),
    ]
    
    all_passed = True
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        print("-"*40)
        if not test_func():
            all_passed = False
    
    print("\n" + "="*60)
    if all_passed:
        print("✓ 所有验证通过！")
        print("\n可以运行: python svg_main.py")
    else:
        print("✗ 存在错误，请检查上述输出")
    
    return all_passed

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)