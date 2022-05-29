import os
import cv2
import time
import numpy as np
from datetime import datetime
from tensorboardX import SummaryWriter
from torchvision.utils import make_grid

BASE_DIR = '/home/duane/Mine/DeepLearning/PytorchPro/mask_rcnn_new/logs'
FILE_DIR = os.path.join(BASE_DIR, 'recorder.txt')

PREDICTION_DIR = '/home/duane/Mine/DeepLearning/PytorchPro/mask_rcnn_new/logs_prediction/'


def save_to_file(contents, file_name=FILE_DIR):
    fh = open(file_name, 'w')
    fh.write(contents)
    fh.close()


# save_to_file(RECORDER_FILE, '123456')

def read_from_file(file=FILE_DIR):
    with open(file) as f:
        line = f.readline().rstrip('\n')
        # print(line)
    return line


# read_from_file(RECORDER_FILE)

def save_p_n_image_to_tensorboard(p_feature_map_list=[], n_feature_map_list=[]):
    log_dir = read_from_file()
    tb_logger = SummaryWriter(log_dir=os.path.join(BASE_DIR, log_dir))

    count = 1
    for feature_map in p_feature_map_list:
        count += 1
        feature_map = feature_map.transpose(0, 1)
        img_grid = make_grid([feature_map[64], feature_map[128]], normalize=True, scale_each=True, nrow=2,
                             padding=20)  # B，C, H, W
        tb_logger.add_image('p' + str(count) + '_feature_maps', img_grid)

    count = 1
    for n_feature_map in n_feature_map_list:
        count += 1
        n_feature_map = n_feature_map.transpose(0, 1)
        img_grid = make_grid([n_feature_map[64], n_feature_map[128]], normalize=True, scale_each=True, nrow=2,
                             padding=20)  # B，C, H, W
        tb_logger.add_image('n' + str(count) + '_feature_maps', img_grid)

    # tb_logger.close()
    # # 绘制原始图像
    # img_raw = normalize_invert(img, normMean, normStd)  # 图像去标准化
    # img_raw = np.array(img_raw * 255).clip(0, 255).squeeze().astype('uint8')
    # writer.add_image('raw img', img_raw, global_step=666)  # j 表示feature map数


def save_image_to_tensorboard(con_name, feature_map_list=[]):
    log_dir = read_from_file()
    tb_logger = SummaryWriter(log_dir=os.path.join(BASE_DIR, log_dir))

    count = 1
    for feature_map in feature_map_list:
        count += 1
        if con_name == 'n' and count == 2:
            continue
        feature_map = feature_map.transpose(0, 1)
        img_grid = make_grid([feature_map[64], feature_map[128]], normalize=True, scale_each=True, nrow=2,
                             padding=20)  # B，C, H, W
        tb_logger.add_image(con_name + str(count) + '_feature_maps', img_grid)

    # tb_logger.close()
    # # 绘制原始图像
    # img_raw = normalize_invert(img, normMean, normStd)  # 图像去标准化
    # img_raw = np.array(img_raw * 255).clip(0, 255).squeeze().astype('uint8')
    # writer.add_image('raw img', img_raw, global_step=666)  # j 表示feature map数


def get_tb_logger():
    log_dir = read_from_file()
    tb_logger = SummaryWriter(log_dir=os.path.join(BASE_DIR, log_dir))
    return tb_logger


def save_img_start(feature_map):
    log_dir = read_from_file()
    tb_logger = SummaryWriter(log_dir=os.path.join(BASE_DIR, log_dir))

    feature_map = feature_map.transpose(0, 1)
    # img_grid = make_grid(feature_map * 255[0], normalize=True, scale_each=True, nrow=3, padding=20)  # B，C, H, W
    # tb_logger.add_image('input_image', img_grid)
    img_grid = make_grid(feature_map, normalize=True, scale_each=True, nrow=3, padding=20)  # B，C, H, W
    tb_logger.add_image('input_feature_maps', img_grid)


##### 预测时的图片
def tb_log_prediction(p_feature_maps, n_feature_maps):
    # timestamp = datetime.fromtimestamp(time.time()).strftime('%Y%m%d-%H:%M')
    # tb_logger = SummaryWriter(os.path.join(PREDICTION_DIR, 'prediction-{}'.format(timestamp)))

    count = 1
    for feature_map in p_feature_maps:
        count += 1
        file_save_dir = os.path.join(PREDICTION_DIR, 'p_feature/' + 'n_' + str(count) + '/')
        if not os.path.exists(file_save_dir):
            os.makedirs(file_save_dir)
        feature_map = feature_map.transpose(0, 1)
        for i in range(len(feature_map)):
            img_mask = np.array(feature_map[i].cpu()) * 255
            img_mask = cv2.merge(img_mask)
            file_name = 'p_f_' + str(count) + 'channel_' + str(i)
            file_path = os.path.join(file_save_dir, file_name + '.png')
            cv2.imwrite(file_path, img_mask)
        # img_grid = make_grid([feature_map[64], feature_map[128]], normalize=True, scale_each=True, nrow=2,
        #                      padding=20)  # B，C, H, W
        # tb_logger.add_image('p' + str(count) + '_feature_maps', img_grid)

    count = 2
    for n_feature_map in n_feature_maps:
        count += 1
        file_save_dir = os.path.join(PREDICTION_DIR, 'n_feature/' + 'n_' + str(count) + '/')
        if not os.path.exists(file_save_dir):
            os.makedirs(file_save_dir)
        n_feature_map = n_feature_map.transpose(0, 1)
        for j in range(len(n_feature_map)):
            img_mask = np.array(n_feature_map[j].cpu()) * 255
            img_mask = cv2.merge(img_mask)
            file_name = 'n_f_' + str(count) + 'channel_' + str(j)
            file_path = os.path.join(file_save_dir, file_name + '.png')
            cv2.imwrite(file_path, img_mask)
        # img_grid = make_grid([n_feature_map[64], n_feature_map[128]], normalize=True, scale_each=True, nrow=2,
        #                     padding=20)  # B，C, H, W
        # tb_logger.add_image('n' + str(count) + '_feature_maps', img_grid)


def tb_log_channel(feature_map, nx_name, name):
    count = 2

    count += 1
    file_save_dir = os.path.join(PREDICTION_DIR, nx_name + '/' + name + '/')
    if not os.path.exists(file_save_dir):
        os.makedirs(file_save_dir)
    feature_map = feature_map.transpose(0, 1)
    for i in range(len(feature_map)):
        img_mask = np.array(feature_map[i].cpu()) * 255
        img_mask = cv2.merge(img_mask)
        file_name = name + str(count) + 'channel_' + str(i)
        file_path = os.path.join(file_save_dir, file_name + '.png')
        cv2.imwrite(file_path, img_mask)
