#!/bin/bash

COMMON="python2 ../../scripts/train/few_shot/run_train.py --data.cuda --data.dataset miniimagenet --data.root $HOME/data/miniimagenet --model.model_name clusternet_conv --iterations 30000"

PYTHONPATH=../.. $COMMON --log.exp_dir results/miniimagenet5.pretrain --train-loss softmax --centroid-loss 0 --data.way 20 --data.test_way 5
