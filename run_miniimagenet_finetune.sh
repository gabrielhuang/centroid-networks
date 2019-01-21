#!/bin/bash

trap "kill 0" SIGINT

COMMON="python2 scripts/train/few_shot/run_train.py --data.cuda --data.dataset miniimagenet --data.root $HOME/data/miniimagenet --model.model_name clusternet_conv --checkpoint-state $HOME/code/cyvius96/save/proto-5/epoch-last.pth"

PYTHONPATH=. $COMMON --log.exp_dir results/miniimagenet/sinkhorn.trainway5.centroid0.01 --train-loss sinkhorn --centroid-loss 0.01 --data.way 5  &
PYTHONPATH=. $COMMON --log.exp_dir results/miniimagenet/sinkhorn.trainway15.centroid0.01 --train-loss sinkhorn --centroid-loss 0.01 --data.way 15 &
PYTHONPATH=. $COMMON --log.exp_dir results/miniimagenet/sinkhorn.trainway30.centroid0.01 --train-loss sinkhorn --centroid-loss 0.01 --data.way 30 &



wait
echo "All runs complete."
