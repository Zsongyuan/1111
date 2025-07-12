# api.py
import os
import io
import requests
import numpy as np
from PIL import Image
from alibabacloud_imageseg20191230.client import Client
from alibabacloud_imageseg20191230.models import SegmentSkinAdvanceRequest
from alibabacloud_tea_util.models import RuntimeOptions
from alibabacloud_tea_openapi.models import Config

def segment_skin(image_path):
    """
    调用阿里云图像分割API，处理皮肤分割，返回处理后图像的URL或错误信息。
    """
    config = Config(
        access_key_id=os.environ.get('ALIBABA_CLOUD_ACCESS_KEY_ID'),
        access_key_secret=os.environ.get('ALIBABA_CLOUD_ACCESS_KEY_SECRET'),
        endpoint='imageseg.cn-shanghai.aliyuncs.com',
        region_id='cn-shanghai'
    )
    runtime_option = RuntimeOptions()
    try:
        if not config.access_key_id or not config.access_key_secret:
            raise ValueError("缺少阿里云Access Key ID或Secret。")
        with open(image_path, 'rb') as img:
            request = SegmentSkinAdvanceRequest(urlobject=img)
            client = Client(config)
            response = client.segment_skin_advance(request, runtime_option)
            response_dict = response.to_map()
            return response_dict
    except Exception as error:
        error_message = f"Error: {str(error)}"
        return error_message

def get_skin_mask(image_path):
    """
    根据提供的 image_path 调用皮肤分割API，并返回二值化的皮肤mask（np.ndarray）。
    如果调用失败，则返回全0的mask。
    """
    # 打开原图以获得尺寸
    try:
        img = Image.open(image_path).convert("RGB")
    except Exception as e:
        raise Exception(f"无法打开图片: {image_path}, 错误: {e}")
    width, height = img.size

    response = segment_skin(image_path)

    # 若响应为错误信息或格式不正确，则返回全0的mask
    if isinstance(response, str) and response.startswith("Error"):
        skin_mask = np.zeros((height, width), dtype=np.uint8)
        return skin_mask
    else:
        try:
            mask_url = response['body']['Data']['URL']
        except KeyError as e:
            skin_mask = np.zeros((height, width), dtype=np.uint8)
            return skin_mask
        else:
            try:
                mask_response = requests.get(mask_url)
                mask_image = Image.open(io.BytesIO(mask_response.content)).convert("L")  # 灰度图
            except Exception as e:
                skin_mask = np.zeros((height, width), dtype=np.uint8)
                return skin_mask

            if mask_image.size != (width, height):
                mask_image = mask_image.resize((width, height), Image.NEAREST)
            skin_mask = np.array(mask_image)
            # 二值化处理：大于128为255，否则为0
            skin_mask = (skin_mask > 128).astype(np.uint8) * 255
            return skin_mask

if __name__ == '__main__':
    # 测试皮肤分割（需要正确配置阿里云API环境变量）
    test_image_path = r"E:\pbn\pbn3\test_out\test_convert_output.png"
    mask = get_skin_mask(test_image_path)
    Image.fromarray(mask).save("skin_mask_test.png")
    print("皮肤mask已保存为 skin_mask_test.png")
