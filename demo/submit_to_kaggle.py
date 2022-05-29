# -*- coding: utf-8 -*-

# --------------------------------------------------
# @Author  : duane
# @File    : img_mask_drawer.py
# @Time    : 2019/7/7 下午11:38
# @Description:
# "Something"
# --------------------------------------------------
"""
Copyright (c) 2019. All rights reserved.
Created by C. L. Wang on 2019/5/17
"""
import os
import pylab
import numpy as np

import skimage.io as io
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection

from pycocotools.coco import COCO



pylab.rcParams['figure.figsize'] = (8.0, 10.0)  # 图片尺寸


def get_center_of_polygon(points):
    """
    获得多边形的中心点
    :param points:
    :return:
    """
    x = [p[0] for p in points]
    y = [p[1] for p in points]
    centroid = [sum(x) / len(points), sum(y) / len(points)]
    print('[Info] 中心点: {}'.format(centroid))
    return centroid


def draw_img(img_data, img_anns, img_id, n_cat, names_of_cats):
    """
    绘制图像
    :param img_data: Img Data
    :param img_anns: Img 标注
    :param img_id: Img ID
    :param n_cat: 类别数量，颜色
    :param names_of_cats: 类别名称，颜色
    :return: 图像
    """

    plt.imshow(img_data)  # 需要提前填充图像

    ax = plt.gca()
    ax.set_autoscale_on(False)

    polygons, color = [], []

    np.random.seed(37)
    color_dict = dict()
    for i in range(n_cat):
        c = (np.random.random((1, 3)) * 0.8 + 0.2).tolist()[0]
        color_dict[i] = c

    cat_text_set = set()
    for ann in img_anns:
        # category_id = int(ann['category_id'].split('_')[0])  # 图像
        category_id = int(ann['category_id'])  # 图像
        category_id = category_id-1
        c = color_dict[category_id]  # 绘制颜色

        if 'segmentation' in ann:
            if type(ann['segmentation']) == list:
                # polygon
                for seg in ann['segmentation']:  # 多边形
                    poly = np.array(seg).reshape((int(len(seg) / 2), 2))
                    c_x, c_y = get_center_of_polygon(poly)  # 计算多边形的中心点
                    c = (np.random.random((1, 3)) * 0.8 + 0.2).tolist()[0]
                    # 0~26是大类别, 其余是小类别 同时 每个标签只绘制一次
                    # if category_id <= 26 and (category_id not in cat_text_set):
                    if category_id not in cat_text_set:
                        tc = c - np.array([0.5, 0.5, 0.5])  # 降低颜色
                        tc = np.maximum(tc, 0.0)  # 最小值0
                        plt.text(c_x, c_y, names_of_cats[category_id], ha='left', wrap=True, color=tc, fontsize=5,
                                 bbox=dict(facecolor='white', alpha=0.5))  # 绘制标签
                        cat_text_set.add(category_id)  # 每个标签只绘制一次

                    polygons.append(pylab.Polygon(poly))  # 绘制多边形
                    color.append(c)

    p = PatchCollection(polygons, facecolor=color, linewidths=0, alpha=0.4)  # 添加多边形
    ax.add_collection(p)
    p = PatchCollection(polygons, facecolor='none', edgecolors=color, linewidths=2)  # 添加多边形的框
    ax.add_collection(p)

    plt.axis('off')
    ax.get_xaxis().set_visible(False)  # this removes the ticks and numbers for x axis
    ax.get_yaxis().set_visible(False)  # this removes the ticks and numbers for y axis

    # plt.show()
    # out_file = '/media/wly/0000678400004823/syj毕业设计/0maskrcnnnew/demo/test_imgs/box_img_syj/' + cn
    out_file = os.path.join('/media/wly/0000678400004823/syj毕业设计/0maskrcnnnew/demo/test_imgs/box_img_syj', 'test_{}.png'.format(img_id))
    plt.savefig(out_file, bbox_inches='tight', pad_inches=0, dpi=200)

    plt.close()  # 避免所有图像绘制在一起

    print('[Info] 绘制图像 {} 完成!'.format(img_id))


def draw_dataset():
    """
    绘制多张图片
    """
    ann_file = os.path.join("/media/wly/0000678400004823/syj毕业设计/0maskrcnnnew/demo", 'test_imgs',
                            'instances_val2014.json')
    prefix_dir = os.path.join("/media/wly/0000678400004823/syj毕业设计/0maskrcnnnew/demo", 'test_imgs/', 'val2014')

    # ann_file = os.path.join(ROOT_DIR, 'datasets', 'instances_train2019.fashion.5.20190521143805.json')
    # prefix_dir = os.path.join(ROOT_DIR, 'datasets', 'test_mini5')

    coco = COCO(ann_file)

    cats = coco.loadCats(coco.getCatIds())
    names_of_cats = [cat['name'] for cat in cats]
    print('[Info] COCO categories: \n{}\n'.format('\n'.join(names_of_cats)))  # 类别

    n_cat = len(names_of_cats)
    print('[Info] 类别数量: {}'.format(n_cat))

    nms_sup = set([cat['supercategory'] for cat in cats])  # 1级类别
    print('[Info] COCO supercategories: \n{}\n'.format('\n'.join(nms_sup)))

    img_ids = coco.getImgIds()
    for img_id in img_ids:
        img_info = coco.loadImgs([img_id])[0]  # 加载图片
        img_data = io.imread(os.path.join(prefix_dir, img_info['file_name']))  # Img

        ann_ids = coco.getAnnIds(imgIds=img_info['id'], iscrowd=None)  # 标注
        anns = coco.loadAnns(ann_ids)

        draw_img(img_data, anns, img_info['id'], n_cat, names_of_cats)


def main():
    draw_dataset()


if __name__ == '__main__':
    main()