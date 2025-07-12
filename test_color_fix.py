# test_color_fix.py
"""
修复后的颜色转换测试
"""
from svg_output import SVGOutput

def test_rgb_to_hex():
    """测试RGB到十六进制转换"""
    output = SVGOutput(None)
    
    test_cases = [
        # (输入, 期望输出)
        ((255, 0, 0), '#ff0000'),      # 纯红色
        ((0, 255, 0), '#00ff00'),      # 纯绿色
        ((0, 0, 255), '#0000ff'),      # 纯蓝色
        ((128.5, 64.3, 192.8), '#8040c1'),  # 浮点数输入：128.5→128, 64.3→64, 192.8→193
        ((255.9, 255.9, 255.9), '#ffffff'),  # 超出范围的浮点数：255.9→255（被限制）
        ((0.1, 0.1, 0.1), '#000000'),        # 接近黑色的浮点数
        ((-5, 300, 128.7), '#00ff81'),       # 边界测试：-5→0, 300→255, 128.7→129
        ((256.0, -1.0, 255.4), '#ff00ff'),   # 更多边界测试：256→255, -1→0, 255.4→255
    ]
    
    print("测试RGB到十六进制转换:")
    print("-" * 50)
    
    all_passed = True
    for rgb_input, expected in test_cases:
        try:
            result = output._rgb_to_hex(rgb_input)
            passed = (result == expected)
            status = "✓" if passed else "✗"
            
            if passed:
                print(f"{status} RGB{rgb_input} -> {result}")
            else:
                print(f"{status} RGB{rgb_input} -> {result} (期望: {expected})")
                all_passed = False
                
        except Exception as e:
            print(f"✗ RGB{rgb_input} -> 错误: {e}")
            all_passed = False
    
    print("-" * 50)
    if all_passed:
        print("✓ 所有测试通过!")
    else:
        print("✗ 存在测试失败!")
    
    return all_passed

def test_color_types():
    """测试颜色类型转换"""
    import numpy as np
    
    print("\n测试颜色类型转换:")
    print("-" * 50)
    
    # 测试不同类型的输入
    test_values = [
        255,          # int
        255.0,        # float
        np.float32(255.0),  # numpy float32
        np.float64(255.0),  # numpy float64
        np.uint8(255),      # numpy uint8
    ]
    
    for val in test_values:
        try:
            # 测试我们修复后的转换逻辑
            converted = max(0, min(255, int(round(float(val)))))
            print(f"✓ {type(val).__name__}({val}) -> int({converted})")
        except Exception as e:
            print(f"✗ {type(val).__name__}({val}) -> 错误: {e}")

def test_edge_cases():
    """测试边界情况"""
    print("\n测试边界情况:")
    print("-" * 50)
    
    output = SVGOutput(None)
    
    edge_cases = [
        # 测试极端值
        ((0, 0, 0), '#000000'),           # 黑色
        ((255, 255, 255), '#ffffff'),     # 白色
        ((-100, -100, -100), '#000000'),  # 负数 -> 0
        ((500, 500, 500), '#ffffff'),     # 超大值 -> 255
        ((127.5, 127.5, 127.5), '#808080'),  # 中间值四舍五入
        ((127.4, 127.6, 127.5), '#7f8080'),  # 混合四舍五入
    ]
    
    all_passed = True
    for rgb_input, expected in edge_cases:
        try:
            result = output._rgb_to_hex(rgb_input)
            passed = (result == expected)
            status = "✓" if passed else "✗"
            
            if passed:
                print(f"{status} RGB{rgb_input} -> {result}")
            else:
                print(f"{status} RGB{rgb_input} -> {result} (期望: {expected})")
                all_passed = False
                
        except Exception as e:
            print(f"✗ RGB{rgb_input} -> 错误: {e}")
            all_passed = False
    
    return all_passed

def run_comprehensive_test():
    """运行综合测试"""
    print("="*60)
    print("SVG颜色转换综合测试")
    print("="*60)
    
    tests = [
        ("基础RGB转换", test_rgb_to_hex),
        ("数据类型转换", test_color_types),
        ("边界情况测试", test_edge_cases),
    ]
    
    all_passed = True
    
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        if test_func == test_color_types:
            test_func()  # 这个测试没有返回值
        else:
            passed = test_func()
            if not passed:
                all_passed = False
    
    print("\n" + "="*60)
    if all_passed:
        print("🎉 所有颜色转换测试通过！")
        print("\n颜色转换功能已修复，可以继续使用系统")
    else:
        print("❌ 部分测试失败，请检查修复代码")
    
    return all_passed

if __name__ == '__main__':
    run_comprehensive_test()