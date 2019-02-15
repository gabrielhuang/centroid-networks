#!/bin/bash

TRAINLOSS=$1

# Replace this accordingly
PATH_TO_PRETRAINED="$HOME/code/cyvius96/save/proto-5/epoch-last.pth"

COMMON="python2 ../../scripts/train/few_shot/run_train.py --data.cuda --data.dataset miniimagenet --data.root $HOME/data/miniimagenet --model.model_name clusternet_conv --iterations 30000"

if [[ "$TRAINLOSS" == "evalonly" ]]
then
	PYTHONPATH=../.. $COMMON --log.exp_dir results/miniimagenet5 --train-loss $TRAINLOSS --centroid-loss 1 --data.way 5
else
	PYTHONPATH=../.. $COMMON --log.exp_dir results/miniimagenet5 --train-loss $TRAINLOSS --centroid-loss 1 --data.way 5 --checkpoint-state $PATH_TO_PRETRAINED
fi

