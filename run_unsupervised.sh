#!/bin/bash
#PYTHONPATH=. python2 scripts/train/few_shot/run_train.py --log.exp_dir results.supervised.sinkhorn.centroid --model.model_name clusternet_conv --mode supervised --supervisedsinkhorn 1 --centroid-loss 1 --data.cuda
PYTHONPATH=. python2 scripts/train/few_shot/run_train.py --log.exp_dir results.supervised.sinkhorn.centroid.20way --model.model_name clusternet_conv --mode supervised --supervisedsinkhorn 1 --centroid-loss 1 --data.cuda --data.way 20 --data.test_way 20
