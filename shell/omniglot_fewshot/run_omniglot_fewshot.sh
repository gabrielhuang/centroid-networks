#!/bin/bash

TRAINLOSS=$1
WAYS=$2
COMMON="python2 ../../scripts/train/few_shot/run_train.py --data.dataset omniglot --data.cuda --model.model_name clusternet_conv"

PYTHONPATH=../.. $COMMON --log.exp_dir results/omniglot${WAYS} --train-loss $TRAINLOSS --centroid-loss 1 --regularization 1 --iterations 100000
