# benchmark_performance.py
"""
性能测试脚本：比较不同模式的处理速度
"""
import time
import os
import sys
from svg_parser import SVGParser
from svg_region_mapper import SVGRegionMapper, FastSVGRegionMapper
import numpy as np
import cv2

def test_region_mapping_performance(svg_file: str, skin_mask: np.ndarray):
    """测试区域映射性能"""
    parser = SVGParser(svg_file)
    elements = parser.parse()
    
    print(f"\n测试文件: {svg_file}")
    print(f"元素数量: {len(elements)}")
    print(f"Mask尺寸: {skin_mask.shape}")
    
    # 测试精确模式
    print("\n精确模式测试...")
    start_time = time.time()
    mapper_accurate = SVGRegionMapper(parser, skin_mask)
    skin_indices_acc, env_indices_acc = mapper_accurate.map_regions()
    accurate_time = time.time() - start_time
    print(f"精确模式耗时: {accurate_time:.2f}秒")
    print(f"皮肤元素: {len(skin_indices_acc)}, 环境元素: {len(env_indices_acc)}")
    
    # 测试快速模式
    print("\n快速模式测试...")
    start_time = time.time()
    mapper_fast = FastSVGRegionMapper(parser, skin_mask)
    skin_indices_fast, env_indices_fast = mapper_fast.map_regions()
    fast_time = time.time() - start_time
    print(f"快速模式耗时: {fast_time:.2f}秒")
    print(f"皮肤元素: {len(skin_indices_fast)}, 环境元素: {len(env_indices_fast)}")
    
    # 性能对比
    speedup = accurate_time / fast_time if fast_time > 0 else 0
    print(f"\n性能提升: {speedup:.1f}倍")
    
    # 结果差异
    skin_diff = abs(len(skin_indices_acc) - len(skin_indices_fast))
    print(f"结果差异: {skin_diff}个元素 ({skin_diff/len(elements)*100:.1f}%)")
    
    return {
        'accurate_time': accurate_time,
        'fast_time': fast_time,
        'speedup': speedup,
        'element_count': len(elements)
    }

def create_test_mask(shape: tuple) -> np.ndarray:
    """创建测试用的皮肤mask"""
    h, w = shape
    mask = np.zeros((h, w), dtype=np.uint8)
    
    # 创建一个椭圆形的皮肤区域
    center_x, center_y = w // 2, h // 2
    axes_x, axes_y = w // 3, h // 3
    
    y, x = np.ogrid[:h, :w]
    mask_ellipse = ((x - center_x) / axes_x) ** 2 + ((y - center_y) / axes_y) ** 2 <= 1
    mask[mask_ellipse] = 255
    
    return mask

def main():
    """运行性能测试"""
    print("="*60)
    print("SVG处理性能测试")
    print("="*60)
    
    # 检查输入目录
    import config
    input_dir = config.CONFIG["INPUT_DIR"]
    
    if not os.path.exists(input_dir):
        print(f"错误: 输入目录不存在: {input_dir}")
        return
    
    svg_files = [f for f in os.listdir(input_dir) if f.lower().endswith('.svg')][:3]  # 最多测试3个文件
    
    if not svg_files:
        print(f"错误: 在{input_dir}目录中未找到SVG文件")
        return
    
    results = []
    
    for svg_file in svg_files:
        svg_path = os.path.join(input_dir, svg_file)
        
        try:
            # 解析SVG获取尺寸
            parser = SVGParser(svg_path)
            parser.parse()
            
            # 创建测试mask
            test_shape = (1000, 1000)  # 使用固定尺寸进行测试
            test_mask = create_test_mask(test_shape)
            
            # 运行测试
            result = test_region_mapping_performance(svg_path, test_mask)
            results.append(result)
            
        except Exception as e:
            print(f"测试文件 {svg_file} 时出错: {e}")
    
    # 汇总结果
    if results:
        print("\n" + "="*60)
        print("性能测试汇总")
        print("="*60)
        
        avg_speedup = np.mean([r['speedup'] for r in results])
        avg_accurate_time = np.mean([r['accurate_time'] for r in results])
        avg_fast_time = np.mean([r['fast_time'] for r in results])
        
        print(f"平均性能提升: {avg_speedup:.1f}倍")
        print(f"精确模式平均耗时: {avg_accurate_time:.2f}秒")
        print(f"快速模式平均耗时: {avg_fast_time:.2f}秒")
        
        print("\n建议：")
        if avg_speedup > 3:
            print("- 快速模式性能提升显著，推荐用于批量处理")
        else:
            print("- 两种模式性能差异不大，可根据质量要求选择")

if __name__ == '__main__':
    main()