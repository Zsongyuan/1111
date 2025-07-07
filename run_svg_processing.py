# run_svg_processing.py
"""
(已简化) SVG数字油画处理系统 - 自动化启动脚本
"""
import sys
import os
import time

def main():
    """
    非交互式主函数：自动以快速模式处理所有输入文件。
    """
    start_time = time.time()
    print("="*60)
    print("SVG数字油画转换系统 - 自动化处理模式")
    print("="*60)

    # 1. 自动设置处理模式为 "fast"
    try:
        import svg_config
        svg_config.PROCESSING_MODE = "fast"
        print("模式：已自动设置为 [快速模式]")
    except ImportError:
        print("错误：无法加载`svg_config.py`，请确保文件存在。")
        return

    # 2. 检查输入/输出目录
    try:
        import config
        input_dir = config.CONFIG["INPUT_DIR"]
        output_dir = config.CONFIG["OUTPUT_DIR"]
    except Exception as e:
        print(f"错误：加载目录配置失败: {e}")
        return

    if not os.path.exists(input_dir):
        print(f"错误: 输入目录不存在: {input_dir}")
        print("请创建input目录并放入需要处理的SVG文件。")
        return
        
    os.makedirs(output_dir, exist_ok=True)

    # 3. 查找并确认要处理的文件
    svg_files = [f for f in os.listdir(input_dir) if f.lower().endswith('.svg')]
    if not svg_files:
        print(f"信息: 在 {input_dir} 目录中未找到任何SVG文件。")
        return
        
    print(f"发现 {len(svg_files)} 个SVG文件，准备开始处理...")

    # 4. 运行主处理程序
    try:
        import svg_main
        # 创建处理器实例并开始批量处理
        processor = svg_main.SVGDigitalPaintingProcessor(use_gpu=True)
        processor.process_folder(input_dir, output_dir, dpi=300)
        
    except KeyboardInterrupt:
        print("\n\n用户中断处理。")
    except Exception as e:
        print(f"\n处理过程中发生严重错误: {e}")
        import traceback
        traceback.print_exc()
        
    # 5. 报告总结
    end_time = time.time()
    print("\n" + "="*60)
    print(f"所有任务已完成，总耗时: {end_time - start_time:.2f} 秒。")
    print(f"输出文件已保存在: {output_dir}")
    print("="*60)


if __name__ == '__main__':
    main()