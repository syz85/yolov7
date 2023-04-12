#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

"""图片预处理"""

from PIL import Image
from annotation_to_yolo_format import WIDTH, HEIGHT, WIDTH_HEIGHT_RATIO
import os


PREDICT_IMAGE_PATH = 'data/seal/predict'


def main():
    target_dir = 'data/seal/raw/img/'
    target_file_name_list = [
        '61.jpeg',
        '62.jpeg',
        '63.jpeg',
        '64.jpeg',
        '65.jpeg',
        '66.jpeg',
        '67.jpeg',
        '68.jpeg',
        '69.jpeg',
        '70.jpeg',
        '71.jpeg',
        '72.jpeg',
        '73.jpeg',
        '74.jpeg',
        '75.jpeg',
        '76.jpeg',
        '77.jpeg',
        '78.jpeg',
        '79.jpeg',
        '80.jpeg',
        '81.jpeg',
        '82.jpeg',
        '83.jpeg',
        '84.jpeg',
        '85.jpeg',
        '86.jpeg',
        '87.jpeg',
        '88.jpeg',
        '89.jpeg',
        '90.jpeg',
    ]

    # 检查文件是否存在
    for file_name in target_file_name_list:
        if not os.path.exists(target_dir + file_name):
            print('文件不存在：' + file_name)
            quit(1)

    # 创建文件夹
    if not os.path.exists(PREDICT_IMAGE_PATH):
        os.makedirs(PREDICT_IMAGE_PATH)

    # 图片填充/缩放到目标尺寸
    for file_name in target_file_name_list:
        file_path = target_dir + file_name

        original_image = Image.open(file_path, 'r')
        original_width = original_image.width
        original_height = original_image.height

        # 将图片转为目标尺寸
        width_height_ratio = original_width / original_height
        if width_height_ratio >= WIDTH_HEIGHT_RATIO:
            # 宽比高要大，按照宽度缩放，然后补充高度
            new_width_without_scale = original_width
            new_height_without_scale = round(new_width_without_scale / WIDTH * HEIGHT)
        else:
            new_height_without_scale = original_height
            new_width_without_scale = round(new_height_without_scale / HEIGHT * WIDTH)

        # 拓展画布
        new_image = Image.new(mode='RGB', size=(new_width_without_scale, new_height_without_scale), color=(255, 255, 255))
        new_image.paste(original_image)
        new_image = new_image.resize(size=(WIDTH, HEIGHT))
        new_image.save(PREDICT_IMAGE_PATH + '/' + file_name)


if __name__ == '__main__':
    main()
