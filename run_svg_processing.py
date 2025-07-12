# run_svg_processing.py
"""
(已更新) SVG数字油画处理系统 - 自动化启动脚本, 包含最终清理功能。
"""
import sys
import os
import time
import glob

def final_cleanup(input_dir: str, output_dir: str):
    """
    在所有处理完成后，清理所有非SVG的临时文件。
    """
    print("\n" + "="*60)
    print("开始执行最终清理...")
    
    # 定义允许保留的文件扩展名
    allowed_extensions = ['.svg']
    
    # 扫描并清理的目录列表
    dirs_to_clean = [input_dir, output_dir]
    
    deleted_count = 0
    
    for directory in dirs_to_clean:
        if not os.path.isdir(directory):
            print(f"警告：目录不存在，跳过清理: {directory}")
            continue
            
        print(f"正在清理目录: {directory}")
        # 使用glob查找所有文件
        all_files = glob.glob(os.path.join(directory, '*'))
        
        for file_path in all_files:
            if os.path.isfile(file_path):
                # 获取文件的扩展名
                file_ext = os.path.splitext(file_path)[1].lower()
                
                # 如果扩展名不在允许列表中，则删除
                if file_ext not in allowed_extensions:
                    try:
                        os.remove(file_path)
                        print(f"  > 已删除临时文件: {os.path.basename(file_path)}")
                        deleted_count += 1
                    except Exception as e:
                        print(f"  > 错误：无法删除文件 {os.path.basename(file_path)}: {e}")

    if deleted_count == 0:
        print("未发现需要清理的临时文件。")
    else:
        print(f"清理完成，共删除了 {deleted_count} 个临时文件。")
    print("="*60)


def main():
    """
    非交互式主函数：自动以快速模式处理所有输入文件，并最终清理临时文件。
    """
    start_time = time.time()
    '''print("="*60)
    print("开始 - 自动化处理模式")
    print("="*60)'''
    
    input_dir, output_dir = "", ""

    try:
        import svg_config
        svg_config.PROCESSING_MODE = "fast"
        # print("模式：已自动设置为 [快速模式]")
    except ImportError:
        print("错误：无法加载`svg_config.py`，请确保文件存在。")
        return

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

    svg_files = [f for f in os.listdir(input_dir) if f.lower().endswith('.svg')]
    if not svg_files:
        print(f"信息: 在 {input_dir} 目录中未找到任何SVG文件。")
        return
        
    print(f"发现 {len(svg_files)} 个SVG文件，准备开始处理...")

    try:
        import svg_main
        processor = svg_main.FacialFeatureProtectedProcessor(use_gpu=True)
        processor.process_folder(input_dir, output_dir, dpi=300)
        
    except KeyboardInterrupt:
        print("\n\n用户中断处理。")
    except Exception as e:
        print(f"\n处理过程中发生严重错误: {e}")
        import traceback
        traceback.print_exc()
        
    # 【核心新增】在所有任务完成后调用清理函数
    final_cleanup(input_dir, output_dir)
    
    end_time = time.time()
    print("\n" + "="*60)
    print(f"所有任务已全部完成，总耗时: {end_time - start_time:.2f} 秒。")
    print(f"输入目录: {input_dir}")
    print(f"输出目录: {output_dir}")
    print("="*60)


if __name__ == '__main__':
    main()