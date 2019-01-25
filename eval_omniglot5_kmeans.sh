
COMMON="python2 scripts/train/few_shot/run_train.py --data.cuda --data.dataset miniimagenet --data.root $HOME/data/miniimagenet --model.model_name clusternet_conv --iterations 1000"

PYTHONPATH=. $COMMON --log.exp_dir results/miniimagenet/kmeansprotonet --train-loss evalonly --data.way 5 --clustering kmeansplusplus --checkpoint-state $HOME/code/cyvius96/save/proto-5/epoch-last.pth --regularization 1 # avoid running many times
