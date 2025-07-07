# svg_config.py
"""
SVG处理系统的配置选项
"""

# 处理模式
PROCESSING_MODE = "fast"  # "fast" 或 "accurate"

# 性能相关配置
REGION_MAPPING_CONFIG = {
    "fast": {
        "use_approximation": True,
        "downsample_factor": 4,
        "batch_size": 100,
        "render_dpi": 75
    },
    "accurate": {
        "use_approximation": False,
        "downsample_factor": 2,
        "batch_size": 50,
        "render_dpi": 150
    }
}

# 调试选项
DEBUG_MODE = False
SAVE_INTERMEDIATE_FILES = False

# 颜色量化配置
QUANTIZATION_CONFIG = {
    "saturation_factor": 1.2,
    "facial_feature_weight": 3.0,
    "boundary_element_weight": 1.5,
    "gpu_batch_size": 1000
}

# 输出配置
OUTPUT_CONFIG = {
    "svg_output": True,
    "png_output": True,
    "comparison_image": True,
    "palette_image": True,
    "output_dpi": 300
}

# 皮肤分割配置
SKIN_SEGMENTATION_CONFIG = {
    "api_max_dimension": 2000,
    "api_max_size_mb": 3,
    "coverage_threshold": 0.5,
    "boundary_range": (0.3, 0.7)
}

# 颜色分配策略
COLOR_DISTRIBUTION_CONFIG = {
    "skin_ratio_threshold": 0.1,
    "min_skin_colors": 5,
    "default_skin_weight": 3,
    "enhanced_skin_weight": 5
}