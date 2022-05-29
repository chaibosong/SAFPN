# -*- coding:utf-8 -*-

""" 
----------------------------------------------------
@file:      submit.py 
@author:    duane
@time:      19-11-29
----------------------------------------------------
@description:

----------------------------------------------------
"""

import pandas as pd


def get_no_rle(length):
    pixel = []
    for i in range(length):
        pixel.append('')

    return pixel


def main():
    csv_path = '/home/duane/Mine/DeepLearning/PytorchPro/mask_rcnn_new/demo/submit_test_jys_output_b_se_sa2_reversed_30_81_new.csv'
    df = pd.read_csv(csv_path)
    df_all_data = pd.read_csv(
        '/home/duane/Mine/DeepLearning/PytorchPro/mask_rcnn_new/demo/submit_test_jys_base_t90_reversed.csv')

    score = 0.991
    df_data = df[df.Score > score]
    df_data.drop(['Score'], axis=1, inplace=True)

    df_not_in = df_all_data[~df_all_data['ImageId'].isin(df_data['ImageId'].tolist())]

    df_no_ship = pd.DataFrame(
        {'ImageId': df_not_in['ImageId'].to_list(), 'EncodedPixels': get_no_rle(len(df_not_in['ImageId'].to_list()))})

    df_submit = pd.concat([df_no_ship, df_data], sort=False)

    df_submit.to_csv('score_{}.csv'.format(score), index=False, sep=',')


if __name__ == '__main__':
    main()
