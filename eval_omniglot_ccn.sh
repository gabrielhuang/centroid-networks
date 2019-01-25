
COMMON="python2 scripts/train/few_shot/run_train.py --data.cuda --data.dataset omniglot_ccn --model.model_name clusternet_conv --iterations 1000"

PYTHONPATH=. $COMMON --log.exp_dir results/omniglot_ccn.centroid0.1 --train-loss evalonly --data.way 5
