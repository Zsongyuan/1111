# getK.py
import re
import os

def extract_k_value(filename):
    """从文件名中提取k值，如果没有找到则返回默认值24"""
    base_name = os.path.basename(filename)
    possible_k_values = {'24', '36', '48'}
    base_name = base_name.split('.')[0]
    parts = re.findall(r'\d+', base_name)

    for part in reversed(parts):
        if part in possible_k_values:
            return int(part)
    return 24  # 默认值

if __name__ == '__main__':
    test_files = ["xxx_24.svg", "yyy_36.svg", "zzz.svg"]
    for f in test_files:
        print(f"{f}: k={extract_k_value(f)}")
