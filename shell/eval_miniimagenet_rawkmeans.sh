
COMMON="python2 scripts/train/few_shot/run_train.py --data.cuda --data.dataset miniimagenet --data.root $HOME/data/miniimagenet --model.model_name clusternet_conv --iterations 1000"

WAYS=5

PYTHONPATH=. $COMMON --log.exp_dir results/miniimagenet$WAYS/kmeans --train-loss evalonly --data.way $WAYS --data.test_way $WAYS --clustering kmeansplusplus --regularization 1 --rawinput 1
