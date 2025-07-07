# svgConvertor.py
import cairosvg
import cv2
import os
import numpy as np

def convert_and_denoise(svg_file, dpi=300):
    """
    将SVG文件转换为PNG格式，根据API限制进行缩放，并对其进行中值滤波去噪。
    返回处理后的图像（np.ndarray）以及生成的原始（未去噪）PNG文件路径。
    """
    png_file_path = os.path.splitext(svg_file)[0] + '.png'
    
    try:
        cairosvg.svg2png(url=svg_file, write_to=png_file_path, dpi=dpi)
    except Exception as e:
        raise Exception(f"SVG到PNG渲染失败: {e}")

    cv_image = cv2.imread(png_file_path, cv2.IMREAD_COLOR)

    if cv_image is None:
        raise Exception(f"图像加载失败: {png_file_path}")

    # --- 修改开始：增加尺寸检查与缩放逻辑 ---

    # 将要被处理的图像（可能是原图，也可能是缩放后的）
    image_to_process = cv_image
    h, w = image_to_process.shape[:2]

    # 定义阿里云API的尺寸限制
    API_MAX_DIM = 2000

    # 检查尺寸是否超出限制
    if h > API_MAX_DIM or w > API_MAX_DIM:
        print(f"  注意: 图像尺寸 ({w}x{h}) 超出API限制 ({API_MAX_DIM}x{API_MAX_DIM})，正在进行等比缩放...")
        
        # 以较长的一边为基准计算缩放比例
        if h > w:
            ratio = API_MAX_DIM / h
        else:
            ratio = API_MAX_DIM / w
            
        new_w = int(w * ratio)
        new_h = int(h * ratio)
        
        # 使用 INTER_AREA 插值进行缩放，这通常是缩小图像的最佳选择
        image_to_process = cv2.resize(image_to_process, (new_w, new_h), interpolation=cv2.INTER_AREA)
        print(f"  缩放后尺寸: {new_w}x{new_h}")
    
    # --- 修改结束 ---

    # 对尺寸合规的图像进行中值滤波
    median_filtered = cv2.medianBlur(image_to_process, 5)
    
    # 函数返回最终处理好的图像数据，以及原始文件路径（主要用于在main.py中清理）
    return median_filtered, png_file_path

if __name__ == '__main__':
    # 测试代码
    # 请确保在项目根目录下有一个名为 'input' 的文件夹，并且里面有SVG文件
    test_svg = os.path.join(".", "input", "CFA-1111-137653-2 40x50 Framed 48.svg")
    output_dir = "test_out"

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if os.path.exists(test_svg):
        print(f"正在测试文件: {test_svg}")
        # 使用一个非常高的DPI来强制触发缩放逻辑
        processed_image, tmp_path = convert_and_denoise(test_svg, dpi=600)
        
        output_path = os.path.join(output_dir, "test_convert_output.png")
        cv2.imwrite(output_path, processed_image)
        
        print(f"转换并去噪后的测试图像已保存到: {output_path}")
        print(f"图像尺寸: {processed_image.shape[1]}x{processed_image.shape[0]}")

        # 清理临时文件
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
    else:
        print(f"测试文件 {test_svg} 不存在")