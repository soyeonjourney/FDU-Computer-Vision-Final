#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python main.py --net=vit --cos --batch-size=256 --aug --max-epoch=200
# CUDA_VISIBLE_DEVICES=0 python main.py --net=vit_small --cos --batch-size=256 --aug --max-epoch=200
# CUDA_VISIBLE_DEVICES=0 python main.py --net=vit_timm --lr=0.0001 --cos --batch-size=128 --amp --max-epoch=10
