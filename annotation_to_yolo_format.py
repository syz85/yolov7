#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

from PIL import Image
import os


"""
图片尺寸：512 x 512
"""
WIDTH = 512
HEIGHT = 512
WIDTH_HEIGHT_RATIO = WIDTH / HEIGHT


_IMAGE_RAW_DIR = 'data/seal/raw/img'
_IMAGE_SAVE_DIR = 'data/seal/images/train'
_LABEL_SAVE_DIR = 'data/seal/labels/train'


def main():
    # 创建文件夹
    if not os.path.exists(_IMAGE_SAVE_DIR):
        os.makedirs(_IMAGE_SAVE_DIR)
    if not os.path.exists(_LABEL_SAVE_DIR):
        os.makedirs(_LABEL_SAVE_DIR)

    with open('data/seal/raw/annotation.txt', 'r', encoding='utf-8') as rf:
        for line in rf:
            line = line.strip('\r\n')
            if not line:
                continue

            fields = line.split(' ')

            # 读取图片
            image_file_name = fields[0]
            image_file_name_prefix = image_file_name[: image_file_name.rindex('.')]
            image_file_path = _IMAGE_RAW_DIR + '/' + image_file_name
            original_image = Image.open(image_file_path, 'r')
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
            new_image.save(_IMAGE_SAVE_DIR + '/' + image_file_name)

            with open(_LABEL_SAVE_DIR + '/' + image_file_name_prefix + '.txt', 'w', encoding='utf-8') as wf:
                for item in fields[1:]:
                    x1, y1, x2, y2, label = item.split(',')
                    x1 = int(x1)
                    x2 = int(x2)
                    y1 = int(y1)
                    y2 = int(y2)
                    x_center = (x2 + x1) / 2
                    y_center = (y2 + y1) / 2
                    w = x2 - x1
                    h = y2 - y1

                    print(f'{label} {x_center / new_width_without_scale} {y_center / new_height_without_scale} {w / new_width_without_scale} {h / new_height_without_scale}', file=wf)


if __name__ == '__main__':
    main()
