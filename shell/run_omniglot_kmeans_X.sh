#!/bin/bash

trap "kill 0" SIGINT

WAYS=$1
COMMON="python2 scripts/train/few_shot/run_train.py --data.cuda --model.model_name clusternet_conv --data.way $WAYS --data.test_way $WAYS"

PYTHONPATH=. $COMMON --log.exp_dir results/omniglot${WAYS}/kmeans --train-loss softmax --clustering kmeansplusplus

wait
echo "All runs complete."
