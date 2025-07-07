# test_color_fix.py
"""
测试颜色转换修复
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
        ((128.5, 64.3, 192.8), '#804cc1'),  # 浮点数输入
        ((255.9, 255.9, 255.9), '#ffffff'),  # 接近白色的浮点数
        ((0.1, 0.1, 0.1), '#000000'),        # 接近黑色的浮点数
    ]
    
    print("测试RGB到十六进制转换:")
    print("-" * 50)
    
    all_passed = True
    for rgb_input, expected in test_cases:
        try:
            result = output._rgb_to_hex(rgb_input)
            passed = (result == expected)
            status = "✓" if passed else "✗"
            print(f"{status} RGB{rgb_input} -> {result} (期望: {expected})")
            if not passed:
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
            int_val = int(val)
            print(f"✓ {type(val).__name__}({val}) -> int({int_val})")
        except Exception as e:
            print(f"✗ {type(val).__name__}({val}) -> 错误: {e}")

if __name__ == '__main__':
    test_rgb_to_hex()
    test_color_types()