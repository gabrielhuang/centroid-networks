# Centroid Networks for Few-shot Clustering and Unsupervised Few-shot Classification

Code for the paper "Centroid Networks for Few-shot Clustering and Unsupervised Few-shot Classification".

The code is forked from the Prototypical Networks code by Jake Snell and collaborators. Thanks to them for sharing their code in the first place! Their instructions (below) have been modified for our project.

## Preparing Data & Dependencies

### Install dependencies

* This code has been tested with Python 2.7 and PyTorch 1.0.
* Install [PyTorch and torchvision](http://pytorch.org/).
* Install [torchnet](https://github.com/pytorch/tnt) by running `pip install torchnet`.
* Install the protonets package by running `python setup.py install` or `python setup.py develop`. Alternatively, call your scripts using the `PYTHONPATH=/PATH/TO/PROTONET/FOLDER python2 /path/to/script.py` syntax.

### Set up the Omniglot dataset

* Run `sh download_omniglot.sh`.

### Set up the MiniImageNet dataset

* Download MiniImageNet from Google Drive: https://drive.google.com/open?id=0B3Irx3uQNoBMQ1FlNXJsZUdYWEE
* Note: If downloading using curl or wget, e.g. on a headless server, follow the trick on stackoverflow: https://stackoverflow.com/a/43816312
* Extract the data to `$HOME/data/miniimagenet/` so that it looks like `$HOME/data/miniimagenet/images/*.jpg`.
* Copy the Ravi splits from the repo to the same folder `cp data/miniImagenet/splits/ravi/*.csv $HOME/dat
a/miniimagenet/`.
* The resulting folder structure should look like
```
$HOME/data/
├──miniimagenet/
  ├── images
	   ├── n0210891500001298.jpg  
	   ├── n0287152500001298.jpg 
	...
  ├── test.csv
  ├── val.csv
  └── train.csv
```

### Third-party MiniImageNet protonets.

We pretrained our models using Protonets, although that is probably not necessary.

Since the upstream repo does not provide code for MiniImageNet, we used a third party implementation.

First, clone the relevant repo in a sibling directory `cyvius96` to our repo:
```
git clone https://github.com/cyvius96/prototypical-network-pytorch.git cyvius96
```

Then, run the training.
```
python train.py --shot 5 --train-way 20 --save-path ./save/proto-5
```

This will save a model in `cyvius96/save/proto-5/epoch-last.pth` which we will reuse in our scripts.

## General arguments

### Scripts

The main script is `scripts/train/few_shot/train_ravioli_free.py`, both for training and evaluation.

The interesting code for few-shot clustering and unsupevised few-shot classification is in
`protonets/models/fewshot.py` and `protonets/models/wasserstein.py`.

For calling from the terminal, we use the wrapper script `scripts/train/few_shot/run_train.py` which takes as arguments
- `--log.exp_dir EXP_DIR`, where `EXP_DIR` is your desired output directory.
- `--data.cuda` for using CUDA or not.


## Results on Omniglot

Go to the folder
```
cd shell/omniglot_fewshot
```

*Train* for 5-shot and 20-shot. This will save results and models in `results/omniglot5` and `results/omniglot20`.
A lot of statistics (runnning averages) such as clustering accuracy are printed during training, but follow procedure below for proper evaluation.
```
./run_omniglot_fewshot.sh sinkhorn 5
./run_omniglot_fewshot.sh sinkhorn 20
```

*Evaluate* for 5-shot and 20-shot. This will create new directories `results/omniglot5.eval` and `results/omniglot20.eval`.
```
./run_omniglot_fewshot.sh evalonly 5
./run_omniglot_fewshot.sh evalonly 20
```

Inspect the resulting logs
```
cat results/omniglot5/summary.txt
cat results/omniglot20/summary.txt
```

The interesting lines are:
- *Test Clustering Accuracy* `test/SupportClusteringAcc_sinkhorn`
- *Test Unsupervised Accuracy* `test/QueryClusteringAcc_sinkhorn`


## Results on MiniImageNet

### Protonet Pretraining (3rd party code)
As of now, our best results are obtained by first pretraining a Prototypical Network to do supervised 20-way classification.
We have used the 3rd party code by [cyvius96](https://github.com/cyvius96/prototypical-network-pytorch)

```
cd $HOME/code  # change path as needed
git clone https://github.com/cyvius96/prototypical-network-pytorch.git cyvius96
python train.py --shot 5 --train-way 20 --save-path ./save/proto-5
```

This creates a pickled checkpoint `$HOME/code/cyvius96/save/proto-5/epoch-last.pth` which we will reuse to initialize Centroid Networks.

### Protonet Pretraining (Experimental!! our code)

Alternatively, pretrain with our code.

Go to the shell script folder
```
cd shell/miniimagenet_fewshot
```

Train for 20-way classification with a softmax loss (reduces to Protonet training). 
```
./pretrain_miniimagenet.sh
```

This creates checkpoints in `/results/miniimagenet5.pretrain/current_model.pt` which can be reused in next step.


### Centroid Networks Fine-tuning

Go to the shell script folder
```
cd shell/miniimagenet_fewshot
```

*Train* for 5-shot. This will save results and models in `results/miniimagenet5`.
A lot of statistics (runnning averages) such as clustering accuracy are printed during training, but follow procedure below for proper evaluation.
```
./run_miniimagenet_fewshot.sh sinkhorn
```

*Evaluate* for 5-shot. This will create a new directory `results/miniimagenet5.eval`.
```
./run_miniimagenet_fewshot.sh evalonly
```

Inspect the resulting logs
```
cat results/miniimagenet5/summary.txt
```

The interesting lines are:
- *Test Clustering Accuracy* `test/SupportClusteringAcc_sinkhorn`
- *Test Unsupervised Accuracy* `test/QueryClusteringAcc_sinkhorn`


## Results on Omniglot (Constrained Clustering Network splits)

This experiment reproduces the setting used in the [Learning to Cluster](https://github.com/GT-RIPL/L2C) projects.

The whole alphabet is fed as one minibatch. It fits on the Nvidia Tesla P100 (16GB ram). If it does not fit on your GPU, you can subsample during training (e.g. use fewer shots), and compute embeddings separately during evaluation.
Specifically for the implementation here, test=validation and support=query for compatibility with the other settings.

Go to the folder
```
cd shell/omniglot_ccn
```

*Train*. This will save results and models in `results/arch_ccn`
```
./run_omniglot_ccn_archccn.sh sinkhorn
```

*Evaluate*. This will create folder `results/arch_ccn.eval` and log results there.
```
./run_omniglot_ccn_archccn.sh evalonly
```

Inspect the resulting logs
```
cat results/omniglot5/summary.txt
```

The interesting lines are:
- *Test Clustering Accuracy* `test/SupportClusteringAcc_sinkhorn`, which can be directly compared with the Omniglot results from the [Learning to Cluster project](https://github.com/GT-RIPL/L2C)
