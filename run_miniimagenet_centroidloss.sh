#!/bin/bash

trap "kill 0" SIGINT

COMMON="python2 scripts/train/few_shot/run_train.py --data.cuda --data.dataset miniimagenet --data.root $HOME/data/miniimagenet --model.model_name clusternet_conv --checkpoint-state $HOME/code/cyvius96/save/proto-5/epoch-last.pth"

PYTHONPATH=. $COMMON --log.exp_dir results/miniimagenet/sinkhorn.centroid0.01 --train-loss sinkhorn --centroid-loss 0.1 &
PYTHONPATH=. $COMMON --log.exp_dir results/miniimagenet/sinkhorn.centroid0.001 --train-loss sinkhorn --centroid-loss 0.001 &
PYTHONPATH=. $COMMON --log.exp_dir results/miniimagenet/sinkhorn.centroid1 --train-loss sinkhorn --centroid-loss 1 &



wait
echo "All runs complete."
