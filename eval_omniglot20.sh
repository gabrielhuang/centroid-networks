
COMMON="python2 scripts/train/few_shot/run_train.py --data.cuda --model.model_name clusternet_conv --iterations 1000"

PYTHONPATH=. $COMMON --log.exp_dir results/omniglot20/sinkhorn.centroid1.20way --train-loss evalonly --data.way 20 --data.test_way 20
