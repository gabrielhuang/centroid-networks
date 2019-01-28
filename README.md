# Centroid Networks for Few-shot Clustering and Unsupervised Few-shot Learning

Code for the paper "Centroid Networks for Few-shot Clustering and Unsupervised Few-shot Learning".

The code is forked from the Prototypical Networks code by Jake Snell and collaborators. Thanks to them for sharing their code in the first place! Their instructions (below) have been modified for our project.

## Training Centroid Networks

### Install dependencies

* This code has been tested with Python 2.7 and PyTorch 1.0.
* Install [PyTorch and torchvision](http://pytorch.org/).
* Install [torchnet](https://github.com/pytorch/tnt) by running `pip install git+https://github.com/pytorch/tnt.git@master`.
* Install the protonets package by running `python setup.py install` or `python setup.py develop`. Alternatively, call your scripts by prepending PYTHONPATH=/PATH/TO/PROTONET/FOLDER before the call to the python interpreter.

### Set up the Omniglot dataset

* Run `sh download_omniglot.sh`.

### Train the model

* Run `python scripts/train/few_shot/run_train.py`. This will run training and place the results into `results`.
  * You can specify a different output directory by passing in the option `--log.exp_dir EXP_DIR`, where `EXP_DIR` is your desired output directory.
  * If you are running on a GPU you can pass in the option `--data.cuda`.
* Re-run in trainval mode `python scripts/train/few_shot/run_trainval.py`. This will save your model into `results/trainval` by default.

### Evaluate

* Run evaluation as: `python scripts/predict/few_shot/run_eval.py --model.model_path results/trainval/best_model.pt`.
