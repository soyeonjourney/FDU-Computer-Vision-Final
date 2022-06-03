import glob
import os

import numpy as np

'''
    0,0 ------> x (width)
     |
     |  (Left,Top)
     |      *_________
     |      |         |
            |         |
     y      |_________|
  (height)            *
                (Right,Bottom)
'''
path = './map_out'
GT_PATH             = os.path.join(path, 'ground-truth')
DR_PATH             = os.path.join(path, 'detection-results')
IMG_PATH            = os.path.join(path, 'images-optional')
TEMP_FILES_PATH     = os.path.join(path, '.temp_files')
RESULTS_FILES_PATH  = os.path.join(path, 'results')

ground_truth_files_list = glob.glob(GT_PATH + '/*.txt')
ground_truth_files_list.sort()
detection_result_files_list = glob.glob(DR_PATH + '/*.txt')
detection_result_files_list.sort()

def compute_IOU(rec1,rec2):
    """
    计算两个矩形框的交并比。
    :param rec1: (x0,y0,x1,y1)      (x0,y0)代表矩形左上的顶点，（x1,y1）代表矩形右下的顶点。下同。
    :param rec2: (x0,y0,x1,y1)
    :return: 交并比IOU.
    """
    for i in range(4):
        rec1[i] = float(rec1[i])
        rec2[i] = float(rec2[i])
    left_column_max  = max(rec1[0],rec2[0])
    right_column_min = min(rec1[2],rec2[2])
    up_row_max       = max(rec1[1],rec2[1])
    down_row_min     = min(rec1[3],rec2[3])
    #两矩形无相交区域的情况
    if left_column_max>=right_column_min or down_row_min<=up_row_max:
        return 0
    # 两矩形有相交区域的情况
    else:
        S1 = (rec1[2]-rec1[0])*(rec1[3]-rec1[1])
        S2 = (rec2[2]-rec2[0])*(rec2[3]-rec2[1])
        S_cross = (down_row_min-up_row_max)*(right_column_min-left_column_max)
        return S_cross/(S1+S2-S_cross)


def get_mIoU():
    total_img = len(ground_truth_files_list)
    sum_stuff = 0
    sum_IoU = 0
    for i in range(total_img):
        GT = open(ground_truth_files_list[i])
        while True:
            gt_line = GT.readline().strip()
            if gt_line == '':
                break
            sum_stuff += 1
            gt_list = gt_line.split(' ')[1:5]
            max_IoU = 0
            DR = open(detection_result_files_list[i])
            while True:
                dr_line = DR.readline().strip()
                if dr_line == '':
                    break
                dr_list = dr_line.split(' ')[2:6]
                IoU = compute_IOU(gt_list, dr_list)
                if max_IoU < IoU:
                    max_IoU = IoU
            sum_IoU += max_IoU
        GT.close()
        DR.close()
    return sum_IoU / sum_stuff

if __name__ == '__main__':
    mIoU = get_mIoU()
    print(mIoU)