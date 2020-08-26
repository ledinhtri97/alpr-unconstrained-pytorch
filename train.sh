#!/bin/bash
#models/eccv-model-scracth
python train-detector.py --model data/lp-detector/wpod-net_pretrained --name my-trained-model --train-dir samples/train-detector --output-dir models/my-trained-model/ -op Adam -lr .001 -its 300000 -bs 1