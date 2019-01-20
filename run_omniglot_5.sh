#!/bin/bash
#PYTHONPATH=. python2 scripts/train/few_shot/run_train.py --log.exp_dir results.supervised.sinkhorn.centroid --model.model_name clusternet_conv --mode supervised --supervisedsinkhorn 1 --centroid-loss 1 --data.cuda
COMMON="python2 scripts/train/few_shot/run_train.py --data.cuda --model.model_name clusternet_conv"

PYTHONPATH=. $COMMON --log.exp_dir results/omniglot5.sinkhorn.centroid0.1 --train-loss sinkhorn --centroid-loss 0.1 & \
PYTHONPATH=. $COMMON --log.exp_dir results/omniglot5.sinkhorn.centroid0.01 --train-loss sinkhorn --centroid-loss 0.01 & \
PYTHONPATH=. $COMMON --log.exp_dir results/omniglot5.sinkhorn.centroid0.001 --train-loss sinkhorn --centroid-loss 0.001 & \
PYTHONPATH=. $COMMON --log.exp_dir results/omniglot5.softmax.centroid0.1 --train-loss softmax --centroid-loss 0.1 & \
PYTHONPATH=. $COMMON --log.exp_dir results/omniglot5.sinkhorn.centroid0.01 --train-loss softmax --centroid-loss 0.01 && fg
