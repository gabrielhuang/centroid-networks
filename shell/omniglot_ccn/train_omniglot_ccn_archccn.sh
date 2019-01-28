#!/bin/bash

COMMON="python2 scripts/train/few_shot/run_train.py --data.dataset omniglot_ccn --data.cuda --model.model_name clusternet_conv"

PYTHONPATH=. $COMMON --log.exp_dir results/omniglot_ccn/arch_protonet --train-loss sinkhorn --centroid-loss 0.1 --regularization 1
