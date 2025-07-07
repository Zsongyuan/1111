# quick_fix.py
"""
快速修复脚本 - 自动应用所有必要的修复
"""
import os
import sys

def apply_config_fixes():
    """应用配置修复"""
    print("应用配置修复...")
    
    try:
        import svg_config
        # 强制使用快速模式
        svg_config.PROCESSING_MODE = "fast"
        # 降低渲染DPI
        svg_config.REGION_MAPPING_CONFIG["fast"]["render_dpi"] = 50
        print("✓ 已设置为快速模式，降低渲染DPI")
    except Exception as e:
        print(f"! 无法修改配置: {e}")

def verify_environment():
    """快速环境验证"""
    print("\n快速环境验证...")
    
    # 检查关键模块
    critical_modules = ['numpy', 'cv2', 'torch', 'svgConvertor', 'api']
    missing = []
    
    for module in critical_modules:
        try:
            __import__(module)
        except ImportError:
            missing.append(module)
    
    if missing:
        print(f"✗ 缺少关键模块: {', '.join(missing)}")
        return False
    else:
        print("✓ 关键模块正常")
        return True

def test_quick_conversion():
    """测试快速转换"""
    print("\n测试颜色转换...")
    
    try:
        from svg_output import SVGOutput
        output = SVGOutput(None)
        
        # 测试问题颜色
        test_color = (255.0, 128.5, 64.3)  # 浮点数
        hex_color = output._rgb_to_hex(test_color)
        print(f"✓ 颜色转换成功: {test_color} -> {hex_color}")
        return True
    except Exception as e:
        print(f"✗ 颜色转换失败: {e}")
        return False

def main():
    """主函数"""
    print("="*60)
    print("SVG数字油画系统 - 快速修复")
    print("="*60)
    
    # 1. 应用配置修复
    apply_config_fixes()
    
    # 2. 验证环境
    if not verify_environment():
        print("\n请先安装缺失的依赖包")
        return False
    
    # 3. 测试转换
    if not test_quick_conversion():
        print("\n颜色转换仍有问题，请检查错误信息")
        return False
    
    print("\n" + "="*60)
    print("✓ 修复完成！")
    print("\n现在可以运行:")
    print("  python run_svg_processing.py")
    print("或")
    print("  python svg_main.py")
    
    return True

if __name__ == '__main__':
    success = main()
    
    if success:
        # 询问是否立即运行
        response = input("\n是否立即运行SVG处理? (y/n): ").strip().lower()
        if response == 'y':
            import subprocess
            subprocess.run([sys.executable, "run_svg_processing.py"])