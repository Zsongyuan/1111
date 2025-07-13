# verify_output_colors.py
"""
验证输出的SVG文件中的颜色是否都存在于预定义的色板中。
"""
import os
from typing import Set, Tuple

# 导入项目模块
import config
from svg_parser import SVGParser

def load_all_palette_colors() -> Set[Tuple[int, int, int]]:
    """
    加载并合并所有环境色板和皮肤色板中的颜色。

    Returns:
        一个包含所有有效颜色（以RGB元组形式）的集合。
    """
    print("正在加载所有色板颜色...")
    
    # 加载环境色板
    env_palette = config.CONFIG["env_palette"]
    
    # 加载所有皮肤色板并合并
    all_skin_colors = []
    skin_palettes = config.CONFIG["skin_palettes"]
    for palette_name, colors in skin_palettes.items():
        all_skin_colors.extend(colors)
        
    # 将所有颜色转换为元组，以便放入集合中进行快速查找
    valid_colors = set(tuple(c) for c in env_palette)
    valid_colors.update(set(tuple(c) for c in all_skin_colors))
    
    print(f"加载完成，共找到 {len(valid_colors)} 种独一无二的有效颜色。")
    return valid_colors


def verify_svg_file(svg_path: str, valid_colors_set: Set[Tuple[int, int, int]]) -> list:
    """
    验证单个SVG文件，返回所有不在色板中的颜色。

    Args:
        svg_path: 要验证的SVG文件路径。
        valid_colors_set: 包含所有有效颜色的集合。

    Returns:
        一个列表，包含所有在SVG中找到但不存在于有效色板中的颜色。
    """
    invalid_colors = []
    try:
        # 使用项目中的SVG解析器
        parser = SVGParser(svg_path)
        elements = parser.parse()
        
        if not elements:
            print(f"  -> 警告: 文件 '{os.path.basename(svg_path)}' 中未找到任何路径元素。")
            return []
            
        # 提取文件中所有独一无二的填充颜色
        svg_colors = set(elem.fill_color for elem in elements if elem.fill_color)
        
        # 检查每种颜色是否在有效色板中
        for color in svg_colors:
            if color not in valid_colors_set:
                invalid_colors.append(color)
                
    except Exception as e:
        print(f"  -> 错误: 解析文件 '{os.path.basename(svg_path)}' 时出错: {e}")
        # 将错误本身视为验证失败
        invalid_colors.append(f"解析错误: {e}")
        
    return invalid_colors


def main():
    """
    主执行函数
    """
    print("="*60)
    print("开始验证输出SVG文件的颜色合规性")
    print("="*60)
    
    # 获取输出目录
    output_dir = config.CONFIG["OUTPUT_DIR"]
    if not os.path.isdir(output_dir):
        print(f"错误: 输出目录 '{output_dir}' 不存在。请先运行处理脚本。")
        return
        
    # 1. 加载所有有效的色板颜色
    valid_colors_set = load_all_palette_colors()
    
    # 2. 找到所有需要验证的SVG文件
    svg_files_to_check = [f for f in os.listdir(output_dir) if f.lower().endswith('.svg')]
    
    if not svg_files_to_check:
        print(f"\n在目录 '{output_dir}' 中未找到任何SVG文件进行验证。")
        return
        
    print(f"\n发现 {len(svg_files_to_check)} 个SVG文件，开始逐一验证...\n")
    
    failed_files = {}
    total_files = len(svg_files_to_check)
    
    # 3. 遍历并验证每个文件
    for i, filename in enumerate(svg_files_to_check):
        print(f"[{i+1}/{total_files}] 正在验证: {filename}")
        svg_path = os.path.join(output_dir, filename)
        
        invalid_colors = verify_svg_file(svg_path, valid_colors_set)
        
        if not invalid_colors:
            print(f"  -> ✓ 通过: 所有颜色均在色板中。")
        else:
            print(f"  -> ✗ 失败: 发现 {len(invalid_colors)} 种非法颜色。")
            failed_files[filename] = invalid_colors
            for color in invalid_colors:
                print(f"    - {color}")
    
    # 4. 打印最终总结报告
    print("\n" + "="*60)
    print("验证总结报告")
    print("="*60)
    
    if not failed_files:
        print("🎉 恭喜！所有SVG文件的颜色均合规！")
    else:
        print(f"验证完成，{len(failed_files)} / {total_files} 个文件未通过验证。")
        print("以下是未通过验证的文件及其包含的非法颜色列表：\n")
        for filename, colors in failed_files.items():
            print(f"文件: {filename}")
            for color in colors:
                print(f"  - {color}")
            print("-" * 20)
            
    print("\n验证结束。")


if __name__ == '__main__':
    main()