#!/bin/bash

COMMON="python2 scripts/train/few_shot/run_train.py data.dataset omniglot_ccn --data.cuda --model.model_name clusternet_conv"

PYTHONPATH=. --log.exp_dir results/debug --data.dataset omniglot_ccn --train-loss sinkhorn --centroid-loss 1 --regularization 1
