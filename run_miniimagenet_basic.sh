#!/bin/bash

PYTHONPATH=. python2 scripts/train/few_shot/run_train.py --log.exp_dir results/miniimagenet/basic --data.dataset miniimagenet --data.root $HOME/data/miniimagenet --model.model_name clusternet_conv --supervisedsinkhorn 1 --centroid-loss 0 --data.cuda --temperature 1
