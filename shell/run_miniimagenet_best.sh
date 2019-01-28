#!/bin/bash

trap "kill 0" SIGINT

COMMON="python2 scripts/train/few_shot/run_train.py --data.cuda --data.dataset miniimagenet --data.root $HOME/data/miniimagenet --model.model_name clusternet_conv --iterations 30000"

#PYTHONPATH=. $COMMON --log.exp_dir results/miniimagenet/sinkhorn.temperature1.centroid0.004 --train-loss sinkhorn --centroid-loss 0.004 --data.way 5  --checkpoint-state $HOME/code/cyvius96/save/proto-5/epoch-last.pth&
#PYTHONPATH=. $COMMON --log.exp_dir results/miniimagenet/sinkhorn.temperature1.centroid0.1 --train-loss sinkhorn --centroid-loss 0.1 --data.way 5  --checkpoint-state $HOME/code/cyvius96/save/proto-5/epoch-last.pth&
#PYTHONPATH=. $COMMON --log.exp_dir results/miniimagenet/sinkhorn.temperature1600.centroid0.004 --train-loss sinkhorn --centroid-loss 0.004 --data.way 5 --checkpoint-state $HOME/code/cyvius96/save/proto-5-temperature1600/epoch-last.pth --temperature 1600 --regularization 160,1600,16000&


#PYTHONPATH=. $COMMON --log.exp_dir results/miniimagenet/sinkhorn.temperature1.centroid0.1 --train-loss softmax --centroid-loss 0.1 --data.way 5  --checkpoint-state $HOME/code/cyvius96/save/proto-5/epoch-last.pth&
PYTHONPATH=. $COMMON --log.exp_dir results/miniimagenet/sinkhorn.temperature1.centroid1 --train-loss sinkhorn --centroid-loss 1 --data.way 5  --checkpoint-state $HOME/code/cyvius96/save/proto-5/epoch-last.pth&
#PYTHONPATH=. $COMMON --log.exp_dir results/miniimagenet/sinkhorn.temperature1.centroid0.1 --train-loss sinkhorn --centroid-loss 0.1 --data.way 5  --checkpoint-state $HOME/code/cyvius96/save/proto-5/epoch-last.pth&
#PYTHONPATH=. $COMMON --log.exp_dir results/miniimagenet/sinkhorn.temperature1.centroid0.01 --train-loss sinkhorn --centroid-loss 0.01 --data.way 5  --checkpoint-state $HOME/code/cyvius96/save/proto-5/epoch-last.pth&
#PYTHONPATH=. $COMMON --log.exp_dir results/miniimagenet/sinkhorn.temperature1.centroid0.001 --train-loss sinkhorn --centroid-loss 0.001 --data.way 5  --checkpoint-state $HOME/code/cyvius96/save/proto-5/epoch-last.pth&

wait
echo "All runs complete."
