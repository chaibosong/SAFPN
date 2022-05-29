# -*- coding:utf-8 -*-

""" 
----------------------------------------------------
@file:      compute_images_mean.py 
@author:    duane
@time:      2019/11/08
----------------------------------------------------
@description:
            compute the pixel mean and std of N images
----------------------------------------------------
"""
import cv2
import random
import numpy as np
import os
from typing import List

def ls_walk_ext_abs(path: str, ext: str) -> List[str]:
        """
        List the files end with ext of the directory at the provided path.
        Args:
            path (str): dir path, ext
            ext (str):
        Returns:
            List[str]: list of files in given path

        """
        file_path_abs = []
        for root, dirs, files in os.walk(path):
            for file in files:
                if file.endswith(ext):
                    file_path_abs.append(os.path.join(root, file))

        return file_path_abs

def compute_mean_std(file_list, num, img_shape):
    """

    Args:
        file_list:
        num:
        img_shape: (h,w)

    Returns:

    """
    means, std = [], []
    img_h, img_w = img_shape
    imgs = np.zeros([img_w, img_h, 3, 1])

    # random choose
    random.shuffle(file_list)

    for idx in range(num):
        print(idx)
        img = cv2.imread(file_list[idx])
        img = cv2.resize(img, (img_h, img_w))
        img = img[:, :, :, np.newaxis]
        imgs = np.concatenate((imgs, img), axis=3)

    imgs = imgs.astype(np.float32) / 255.

    for i in range(3):
        pixels = imgs[:, :, i, :].ravel()
        means.append(np.mean(pixels))
        std.append(np.std(pixels))

    # BGR -> RGB
    means.reverse()
    std.reverse()

    print("normMean = {}".format(means))
    print("normStd = {}".format(std))
    print('transforms.Normalize(normMean = {}, normStd = {})'.format(means, std))

    return means, std


if __name__ == '__main__':
    root_dir = '/home/duane/Mine/DeepLearning/PytorchPro/maskrcnn-benchmark/datasets/coco/train2014/'
    file_list = ls_walk_ext_abs(root_dir, '.jpg')
    compute_mean_std(file_list, 2000, (224, 224))

    #################################
    # random 100
    # normMean = [0.49146822, 0.35444292, 0.29199433]
    # normStd = [0.19413905, 0.14409325, 0.128972]
    #
    # random 200
    # normMean = [0.5023309, 0.3591933, 0.29787418]
    # normStd = [0.18451121, 0.1369749, 0.1255672]
    #################################
