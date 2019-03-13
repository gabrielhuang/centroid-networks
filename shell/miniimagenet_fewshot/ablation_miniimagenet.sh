#!/bin/bash

MODE=$1  # train/eval
CLUSTERING=$2  # kmeansplusplus/wasserstein
TRAINLOSS=$3  # softmax/sinkhorn
CENTERLOSS=$4  #  0/1

DIR="ablation/CLUSTER$CLUSTERING-TRAIN$TRAINLOSS-CENTER$CENTERLOSS"
echo "Will output results in directory $DIR"

# Replace this accordingly
PATH_TO_PRETRAINED="$HOME/code/cyvius96/save/proto-5/epoch-last.pth"

COMMON="python2 ../../scripts/train/few_shot/run_train.py --data.cuda --data.dataset miniimagenet --data.root $HOME/data/miniimagenet --model.model_name clusternet_conv --iterations 30000 --data.way 5"

if [[ "$MODE" == "eval" ]]
then
	PYTHONPATH=../.. $COMMON --log.exp_dir $DIR --train-loss evalonly --centroid-loss $CENTERLOSS --clustering $CLUSTERING
else
	PYTHONPATH=../.. $COMMON --log.exp_dir $DIR --train-loss $TRAINLOSS --checkpoint-state $PATH_TO_PRETRAINED --centroid-loss $CENTERLOSS --clustering $CLUSTERING
fi

