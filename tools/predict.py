#--coding:utf-8--


# import matplotlib.pyplot as plt
# import matplotlib.pylab as pylab

# from PIL import Image
# import numpy as np


# from maskrcnn_benchmark.config import cfg
# from maskrcnn_benchmark.predictor import ShipsDemo

# # ���������ļ�
# config_file = r"/home/featurize/data/mask_rcnn_new/tools/admy_mask_rcnn_R_101_FPN_1x.yaml"

# # update the config options with the config file
# cfg.merge_from_file(config_file)
# # manual override some options
# cfg.merge_from_list(["MODEL.DEVICE", "cuda"])
# # cfg.MODEL.WEIGHT = '../pretrained/e2e_mask_rcnn_R_101_FPN_1x.pth'

# coco_demo = ShipsDemo(cfg, min_image_size=800, confidence_threshold=0.7, )

# # if False:
# #     pass
# # else:
# #imgurl = "http://farm3.staticflickr.com/2469/3915380994_2e611b1779_z.jpg"
# # response = requests.get(imgurl)
# # pil_image = Image.open(BytesIO(response.content)).convert("RGB")

# imgfile = r'/home/featurize/data/mask_rcnn_new/tools/in/6.jpg'

# pil_image = Image.open(imgfile).convert("RGB")
# pil_image.show()
# image = np.array(pil_image)[:, :, [2, 1, 0]]
# # print(image)
# # forward predict
# coco_demo.run_on_opencv_image(pil_image)
# # plt.subplot(1, 2, 1)
# # plt.imshow(image[:, :, ::-1])
# # plt.axis('off')

# # plt.subplot(1, 2, 2)
# # plt.imshow(predictions[:, :, ::-1])
# # plt.axis('off')
# # plt.show()



# -*- coding: utf-8 -*-
# @Author  : MengYangD
# @FileName: demo_test.py

from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.predictor import COCODemo
import cv2
import os


config_file = "/media/wly/0000678400004823/syj毕业设计/0maskrcnnnew/tools/admy_mask_rcnn_R_101_FPN_1x.yaml"


# update the config options with the config file
cfg.merge_from_file(config_file)
# manual override some options
cfg.merge_from_list(["MODEL.DEVICE", "cuda"])

ships_demo = COCODemo(
    cfg,
    min_image_size=400,
    confidence_threshold=0.85,
)
# load image and then run prediction


base_img_dir = '/media/wly/0000678400004823/syj毕业设计/0maskrcnnnew/tools/'
save_dir = os.path.join(base_img_dir, 'out')
image_dir = os.path.join(base_img_dir, 'imagestest')
count = 0
#
# for img_file in os.listdir(image_dir):
# #     print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
#
# #     print(str(count) + 'th:' 'Demo for data/demo/{}'.format(img_file))
#     img = cv2.imread(os.path.join(image_dir, img_file))
#     print(img_file,"--------------------")
# #     print(img_file.replace('jpg','txt'))
#     img_file=os.path.splitext(img_file)[0]+'.txt'
#     ships_demo.run_on_opencv_image(img,img_file)
# #     print(predictions)
# #     top_predictions = ships_demo.select_top_predictions(predictions)
#
# #     # box
# #     result = img.copy()
# #     result = ships_demo.overlay_boxes(result, top_predictions)  # �߿�
# #     result = ships_demo.overlay_class_names(result, top_predictions)  # ����
# #     show_mask(result, top_predictions, ships_demo, save_img_name=img_file, save_dir=save_dir)  # mask ����
#     count += 1
# print("cout-----------:",count)


count=0
for img_file in os.listdir(image_dir):
    img_file=os.path.splitext(img_file)[0]+'.txt'
    if not os.path.exists("out/"+img_file):
           count+=1
           with open("out/"+img_file,'a') as f:
                f.write('')
print("cout-----------:",count)