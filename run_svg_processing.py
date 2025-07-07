# run_svg_processing.py
"""
SVG数字油画处理系统启动脚本
提供简单的命令行界面选择处理模式
"""
import sys
import os

def main():
    print("="*60)
    print("SVG数字油画转换系统")
    print("="*60)
    
    # 选择处理模式
    print("\n请选择处理模式:")
    print("1. 快速模式 (推荐用于大文件或批量处理)")
    print("2. 精确模式 (推荐用于高质量输出)")
    print("3. 退出")
    
    while True:
        choice = input("\n请输入选项 (1/2/3): ").strip()
        
        if choice == '3':
            print("退出程序")
            return
            
        if choice in ['1', '2']:
            break
        else:
            print("无效选项，请重新输入")
    
    # 更新配置
    mode = "fast" if choice == '1' else "accurate"
    
    # 动态修改配置
    import svg_config
    svg_config.PROCESSING_MODE = mode
    
    print(f"\n已选择: {'快速模式' if mode == 'fast' else '精确模式'}")
    
    # 检查输入目录
    import config
    input_dir = config.CONFIG["INPUT_DIR"]
    
    if not os.path.exists(input_dir):
        print(f"\n错误: 输入目录不存在: {input_dir}")
        print("请创建input目录并放入SVG文件")
        return
    
    svg_files = [f for f in os.listdir(input_dir) if f.lower().endswith('.svg')]
    
    if not svg_files:
        print(f"\n错误: 在{input_dir}目录中未找到SVG文件")
        return
    
    print(f"\n找到 {len(svg_files)} 个SVG文件:")
    for f in svg_files[:5]:  # 只显示前5个
        print(f"  - {f}")
    if len(svg_files) > 5:
        print(f"  ... 还有 {len(svg_files) - 5} 个文件")
    
    # 确认处理
    confirm = input("\n是否开始处理? (y/n): ").strip().lower()
    
    if confirm != 'y':
        print("已取消处理")
        return
    
    # 运行主程序
    print("\n开始处理...")
    print("-"*60)
    
    try:
        import svg_main
        svg_main.main()
        print("\n处理完成!")
        
    except KeyboardInterrupt:
        print("\n\n用户中断处理")
    except Exception as e:
        print(f"\n处理出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()