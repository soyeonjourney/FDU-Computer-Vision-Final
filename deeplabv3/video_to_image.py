#! /usr/bin/env python3
# -*- coding:utf-8 -*-
# __author__ = "Tqla_Snrs"


import cv2


def video_to_image(src):
    """
    Convert video frame to JPEG file.
    """
    vc = cv2.VideoCapture('./' + src)
    if vc.isOpened():
        rval, frame = vc.read()
    else:
        rval = False

    count = 0

    while rval:
        rval, frame = vc.read()
        if rval:
            # Save frame as JPEG file.
            cv2.imwrite('./data/thn/' + str(count) + '.jpg', frame)

            count += 1

            cv2.waitKey(1)

    vc.release()
