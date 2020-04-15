#!/bin/bash

TRAINLOSS=$1
CLUSTERING=${2:-wasserstein}
COMMON="python2 ../../scripts/train/few_shot/run_train.py --data.dataset omniglot_ccn --data.cuda --model.model_name clusternet_conv"

PYTHONPATH=../.. $COMMON --log.exp_dir results/arch_protonet_$CLUSTERING --train-loss $TRAINLOSS --centroid-loss 0.1 --regularization 1 --iterations 10000 --clustering $CLUSTERING
