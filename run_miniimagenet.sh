#!/bin/bash

#PYTHONPATH=. python2 scripts/train/few_shot/run_train.py --log.exp_dir miniimagenet.supervised.sinkhorn.centroid --data.dataset miniimagenet --data.root $HOME/data/miniimagenet --model.model_name clusternet_conv --supervisedsinkhorn 1 --centroid-loss 1 --checkpoint-state $HOME/code/cyvius96/save/proto-5/epoch-last.pth --data.cuda

# THis run normalizes the temperature/regularization of 1600.
PYTHONPATH=. python2 scripts/train/few_shot/run_train.py --log.exp_dir miniimagenet.supervised.sinkhorn.centroid --data.dataset miniimagenet --data.root $HOME/data/miniimagenet --model.model_name clusternet_conv --supervisedsinkhorn 1 --centroid-loss 0 --checkpoint-state $HOME/code/cyvius96/save/proto-5-temperature1600/epoch-last.pth --data.cuda --temperature 1600 #600
