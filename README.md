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

## General arguments

### Scripts

The main script is `scripts/train/few_shot/train_ravioli_free.py`, both for training and evaluation.

The interesting code for few-shot clustering and unsupevised few-shot classification is in
`protonets/models/fewshot.py` and `protonets/models/wasserstein.py`.

For calling from the terminal, we use the wrapper script `scripts/train/few_shot/run_train.py` which takes as arguments
- `--log.exp_dir EXP_DIR`, where `EXP_DIR` is your desired output directory.
- `--data.cuda` for using CUDA or not.

### Evaluate

* Run evaluation as: `python scripts/predict/few_shot/run_eval.py --model.model_path results/trainval/best_model.pt`.

## Results on Omniglot

## Results on MiniImageNet

## Results on Omniglot (Constrained Clustering Network splits)

This experiment reproduces the setting used in the [Learning to Cluster](https://github.com/GT-RIPL/L2C) projects.

The whole alphabet is fed as one minibatch. It fits on the Nvidia Tesla P100 (16GB ram). If it does not fit on your GPU, you can subsample during training (e.g. use fewer shots), and compute embeddings separately during evaluation.
Specifically for the implementation here, test=validation and support=query for compatibility with the other settings.

Go to the folder
```
cd shell/omniglot_ccn
```

*Train*. This will save results and models in results/arch_ccn
```
./run_omniglot_ccn_archccn.sh sinkhorn
```

*Evaluate*. This will create folder results/arch_ccn.eval and log results there.
```
./run_omniglot_ccn_archccn.sh evalonly
```
