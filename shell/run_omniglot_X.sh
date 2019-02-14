#!/bin/bash

trap "kill 0" SIGINT

WAYS=$1
COMMON="python2 scripts/train/few_shot/run_train.py --data.cuda --model.model_name clusternet_conv --data.way $WAYS --data.test_way $WAYS"

(PYTHONPATH=. $COMMON --log.exp_dir results/omniglot${WAYS}/sinkhorn.centroid0.1 --train-loss sinkhorn --centroid-loss 0.1) & \
(PYTHONPATH=. $COMMON --log.exp_dir results/omniglot${WAYS}/sinkhorn.centroid0.01 --train-loss sinkhorn --centroid-loss 0.01)& \
(PYTHONPATH=. $COMMON --log.exp_dir results/omniglot${WAYS}/sinkhorn.centroid0.001 --train-loss sinkhorn --centroid-loss 0.001)& \
(PYTHONPATH=. $COMMON --log.exp_dir results/omniglot${WAYS}/softmax.centroid0.1 --train-loss softmax --centroid-loss 0.1 )& \
(PYTHONPATH=. $COMMON --log.exp_dir results/omniglot${WAYS}/softmax.centroid0.01 --train-loss softmax --centroid-loss 0.01)& \
(PYTHONPATH=. $COMMON --log.exp_dir results/omniglot${WAYS}/softmax.centroid0.001 --train-loss softmax --centroid-loss 0.001)

wait
echo "All runs complete."
