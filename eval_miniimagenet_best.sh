
COMMON="python2 scripts/train/few_shot/run_train.py --data.cuda --data.dataset miniimagenet --data.root $HOME/data/miniimagenet --model.model_name clusternet_conv --iterations 1000"

PYTHONPATH=. $COMMON --log.exp_dir results/miniimagenet/sinkhorn.temperature1.centroid1 --train-loss evalonly --data.way 5
