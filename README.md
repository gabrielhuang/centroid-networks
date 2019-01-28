# Centroid Networks for Few-shot Clustering and Unsupervised Few-shot Learning

Code for the paper "Centroid Networks for Few-shot Clustering and Unsupervised Few-shot Learning".

The code is forked from the Prototypical Networks code by Jake Snell and collaborators. Thanks to them for sharing their code in the first place! Their instructions (below) have been modified for our project.

## Preparing Data & Dependencies

### Install dependencies

* This code has been tested with Python 2.7 and PyTorch 1.0.
* Install [PyTorch and torchvision](http://pytorch.org/).
* Install [torchnet](https://github.com/pytorch/tnt) by running `pip install git+https://github.com/pytorch/tnt.git@master`.
* Install the protonets package by running `python setup.py install` or `python setup.py develop`. Alternatively, call your scripts by prepending PYTHONPATH=/PATH/TO/PROTONET/FOLDER before the call to the python interpreter.

### Set up the Omniglot dataset

* Run `sh download_omniglot.sh`.

### Set up the MiniImageNet dataset

* Download MiniImageNet from Google Drive: https://drive.google.com/open?id=0B3Irx3uQNoBMQ1FlNXJsZUdYWEE
* Note: If downloading using curl or wget, e.g. on a headless server, follow the trick on stackoverflow: https://stackoverflow.com/a/43816312
* Extract the data to `$HOME/data/miniimagenet/` so that it looks like `$HOME/data/miniimagenet/images/*.jpg`.
* Copy the splits from the repo to the same folder `cp data/miniImagenet/splits/ravi/*.csv $HOME/dat
a/miniimagenet/`.

## General arguments

### Train the model

* Run `python scripts/train/few_shot/run_train.py`. This will run training and place the results into `results`.
  * You can specify a different output directory by passing in the option `--log.exp_dir EXP_DIR`, where `EXP_DIR` is your desired output directory.
  * If you are running on a GPU you can pass in the option `--data.cuda`.
* Re-run in trainval mode `python scripts/train/few_shot/run_trainval.py`. This will save your model into `results/trainval` by default.

### Evaluate

* Run evaluation as: `python scripts/predict/few_shot/run_eval.py --model.model_path results/trainval/best_model.pt`.

## Results on Omniglot

## Results on MiniImageNet

## Results on Omniglot (Constrained Clustering Network splits)
