#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

import torch
from PIL import Image
import numpy as np
from annotation_to_yolo_format import WIDTH, HEIGHT, WIDTH_HEIGHT_RATIO
from utils.datasets import letterbox
from utils.general import non_max_suppression
import cv2


def main():
    # original_image = Image.open('/home/sunyuanzhen/workspace/yolov7/data/seal/raw/img/1.jpeg')
    # original_width = original_image.width
    # original_height = original_image.height
    # # 将图片转为目标尺寸
    # width_height_ratio = original_width / original_height
    # if width_height_ratio >= WIDTH_HEIGHT_RATIO:
    #     # 宽比高要大，按照宽度缩放，然后补充高度
    #     new_width_without_scale = original_width
    #     new_height_without_scale = round(new_width_without_scale / WIDTH * HEIGHT)
    # else:
    #     new_height_without_scale = original_height
    #     new_width_without_scale = round(new_height_without_scale / HEIGHT * WIDTH)
    #
    # # 拓展画布
    # new_image = Image.new(mode='RGB', size=(new_width_without_scale, new_height_without_scale), color=(255, 255, 255))
    # new_image.paste(original_image)
    # new_image = new_image.resize(size=(WIDTH, HEIGHT))
    #
    # img = np.array(new_image, dtype=np.float32)

    model = torch.jit.load('runs/train/yolov7_seal/weights/best.torchscript.pt')
    img = cv2.imread('/home/sunyuanzhen/workspace/yolov7/data/seal/images/train/1.jpeg')
    img = np.array(img, dtype=np.float32)

    img = img[:, :, ::-1].transpose(2, 0, 1)
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img)
    img /= 255.0
    img = img.unsqueeze(0)

    with torch.no_grad():
        predict = model(img)
    predict = predict[0]

    """
    tensor([[[8.09375e+00, 7.39062e+00, 1.73125e+01, 1.24297e+01, 7.62939e-06, 9.83398e-01],
         [1.30625e+01, 9.24219e+00, 3.03281e+01, 1.81562e+01, 1.80960e-04, 9.83398e-01],
         [1.80938e+01, 7.14062e+00, 2.80625e+01, 1.28906e+01, 3.11732e-05, 9.83398e-01],
         ...,
         [4.44250e+02, 5.27000e+02, 6.01562e-01, 1.00977e+00, 1.13249e-06, 9.84863e-01],
         [4.76250e+02, 5.27000e+02, 5.94238e-01, 1.00977e+00, 1.13249e-06, 9.84863e-01],
         [5.08000e+02, 5.27000e+02, 7.13379e-01, 1.18848e+00, 1.07288e-06, 9.84863e-01]]], device='cuda:0', dtype=torch.float16)
    """

    pred = non_max_suppression(predict, 0.80, 0.45, classes=None, agnostic=False)
    print(pred)
    for i, det in enumerate(pred):
        print(det)


if __name__ == '__main__':
    main()
