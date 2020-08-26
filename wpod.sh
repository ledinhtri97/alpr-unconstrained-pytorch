#!/bin/bash
#/workspace/dataset/lpd_poly/lpd_poly_groundtruth_train.txt \
python train-wpod.py -rd /workspace/dataset/lpd_poly \
-gt /workspace/dataset/lpd_poly/lpd_poly_groundtruth_train.txt \
-bs 32 \
-lr 0.0001 \
-pt wpod_cvt.pth