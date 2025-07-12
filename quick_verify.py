# quick_verify.py
"""
快速验证颜色转换修复
"""
import sys

def manual_test_rgb_conversion():
    """手动测试RGB转换逻辑"""
    print("手动验证RGB转换逻辑:")
    print("-" * 40)
    
    # 复制修复后的转换逻辑
    def safe_rgb_to_hex(rgb):
        r = max(0, min(255, int(round(float(rgb[0])))))
        g = max(0, min(255, int(round(float(rgb[1])))))
        b = max(0, min(255, int(round(float(rgb[2])))))
        return f'#{r:02x}{g:02x}{b:02x}'
    
    test_cases = [
        ((255, 0, 0), '#ff0000'),
        ((128.5, 64.3, 192.8), '#8040c1'),  # 128, 64, 193
        ((255.9, 255.9, 255.9), '#ffffff'), # 限制到255
        ((0.1, 0.1, 0.1), '#000000'),
        ((-5, 300, 128.7), '#00ff81'),      # 0, 255, 129
    ]
    
    all_passed = True
    for rgb_input, expected in test_cases:
        result = safe_rgb_to_hex(rgb_input)
        passed = (result == expected)
        status = "✓" if passed else "✗"
        
        if passed:
            print(f"{status} RGB{rgb_input} -> {result}")
        else:
            print(f"{status} RGB{rgb_input} -> {result} (期望: {expected})")
            all_passed = False
    
    return all_passed

def test_import_and_function():
    """测试导入和函数调用"""
    print("\n测试模块导入和函数调用:")
    print("-" * 40)
    
    try:
        from svg_output import SVGOutput
        print("✓ 成功导入 SVGOutput")
        
        # 创建实例（传入None作为parser）
        output = SVGOutput(None)
        print("✓ 成功创建 SVGOutput 实例")
        
        # 测试颜色转换函数
        test_color = (128.5, 64.3, 192.8)
        result = output._rgb_to_hex(test_color)
        expected = '#8040c1'
        
        if result == expected:
            print(f"✓ 颜色转换正确: {test_color} -> {result}")
            return True
        else:
            print(f"✗ 颜色转换错误: {test_color} -> {result} (期望: {expected})")
            return False
            
    except Exception as e:
        print(f"✗ 导入或调用失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主验证函数"""
    print("="*50)
    print("颜色转换修复快速验证")
    print("="*50)
    
    # 测试1: 手动验证转换逻辑
    test1_passed = manual_test_rgb_conversion()
    
    # 测试2: 验证实际模块功能
    test2_passed = test_import_and_function()
    
    # 总结
    print("\n" + "="*50)
    print("验证结果汇总:")
    print(f"  转换逻辑: {'✓ 通过' if test1_passed else '✗ 失败'}")
    print(f"  模块功能: {'✓ 通过' if test2_passed else '✗ 失败'}")
    
    if test1_passed and test2_passed:
        print("\n🎉 颜色转换修复成功！")
        print("\n现在可以运行:")
        print("  python test_color_fix.py    # 运行完整测试")
        print("  python validate_fixes.py    # 验证系统修复")
        print("  python svg_main.py          # 开始处理SVG")
        return True
    else:
        print("\n❌ 修复验证失败，请检查代码")
        return False

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)