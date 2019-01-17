import os
import json
import time
from collections import OrderedDict

import numpy as np

import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torchvision
import torchnet as tnt

import sys
sys.path.append('../../../')

from protonets.engine import Engine

import protonets.utils.data as data_utils
import protonets.utils.model as model_utils
import protonets.utils.log as log_utils


###########################################
# Utils
###########################################
class Summary(object):
    def __init__(self):
        self.logs = OrderedDict()

    def log(self, epoch, name, value):
        self.logs.setdefault(name, {})
        self.logs[name][epoch] = value

    def sorted(self):
        sorted_logs = OrderedDict()
        for log in self.logs:
            sorted_logs[log] = list(sorted(self.logs[log].items()))
        return sorted_logs

    def print_summary(self, n_avg=50):
        sorted_logs = self.sorted()
        print 'Summary'
        for log in sorted_logs:
            tail = sorted_logs[log]
            tail = tail[-min(len(tail), n_avg):]
            val = dict(tail).values()
            print '\t{}: {:.4f} +/- {:.4f}'.format(log, np.mean(val), np.std(val))

class Timer:
    def __enter__(self):
        self.start = time.clock()
        return self

    def __exit__(self, *args):
        self.end = time.clock()
        self.interval = self.end - self.start

def make_infinite(iterator):
    while True:
        new_epoch = True
        for x in iterator:
            yield x, new_epoch
            new_epoch = False


def main(opt):
    ###########################################
    # Boilerplate
    ###########################################
    if not os.path.isdir(opt['log.exp_dir']):
        os.makedirs(opt['log.exp_dir'])

    # save opts
    with open(os.path.join(opt['log.exp_dir'], 'opt.json'), 'w') as f:
        json.dump(opt, f, indent=4)
        f.write('\n')

    trace_file = os.path.join(opt['log.exp_dir'], 'trace.txt')

    # Postprocess arguments
    opt['model.x_dim'] = list(map(int, opt['model.x_dim'].split(',')))
    opt['log.fields'] = opt['log.fields'].split(',')

    torch.manual_seed(1234)
    if opt['data.cuda']:
        torch.cuda.manual_seed(1234)

    ###########################################
    # Data
    ###########################################
    if opt['data.trainval']:
        data = data_utils.load(opt, ['trainval'])
        train_loader = data['trainval']
        val_loader = None

        # Prepare datasets
        train_iter = make_infinite(train_loader)
        val_iter = None
    else:
        data = data_utils.load(opt, ['train', 'val'])
        train_loader = data['train']
        val_loader = data['val']

        # Prepare datasets
        train_iter = make_infinite(train_loader)
        other_train_iter = make_infinite(train_loader)  # for evaluating other losses
        val_iter = make_infinite(val_loader)

    ###########################################
    # Create model and optimizer
    ###########################################
    model = model_utils.load(opt)

    if opt['checkpoint']:
        print 'Loading from checkpoint', opt['checkpoint']
        model = torch.load(opt['checkpoint'])

    if opt['data.cuda']:
        model.cuda()

    Optimizer = getattr(optim, opt['train.optim_method'])
    optimizer = Optimizer(model.parameters(), lr=opt['train.learning_rate'], weight_decay=opt['train.weight_decay'])

    scheduler = lr_scheduler.StepLR(optimizer, opt['train.decay_every'], gamma=0.5)

    ###########################################
    # Training loop
    ###########################################

    summary = Summary()

    #### Start of training loop
    regularization = 1. / opt['temperature']
    for iteration in xrange(opt['iterations']):

        # Sample from training
        with Timer() as train_load_timer:

            sample_train, new_epoch = train_iter.next()

        # Compute loss; backprop
        with Timer() as train_backprop_timer:

            # z = h(x)
            embedding_train = model.embed(sample_train, raw_input=opt['rawinput'])

            # Supervised and Clustering Losses
            supervised_loss, train_supervised_info = model.supervised_loss(embedding_train, regularization=regularization, supervised_sinkhorn_loss=opt['supervisedsinkhorn'])
            __, train_clustering_info = model.clustering_loss(embedding_train, regularization=regularization, supervised_sinkhorn_loss=opt['supervisedsinkhorn'])

            total_loss = supervised_loss

            if opt['centroid_loss'] > 0. :
                centroid_loss = opt['centroid_loss'] * train_supervised_info['ClassVariance']
                total_loss = total_loss + centroid_loss
                summary.log(iteration, 'train/CentroidLoss', centroid_loss.item())  # Supervised accuracy
            summary.log(iteration, 'train/CentroidLossUnscaled', train_supervised_info['ClassVariance'].item())  # Supervised accuracy

            if not opt['rawinput']:
                # No need to backprop in rawinput mode
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

        summary.log(iteration, 'train/SupervisedAcc', train_supervised_info['SupervisedAccuracy'])
        summary.log(iteration, 'train/SupervisedLoss', train_supervised_info['SupervisedLoss'])
        summary.log(iteration, 'train/SupportClusteringAcc', train_clustering_info['SupportClusteringAcc'])
        summary.log(iteration, 'train/QueryClusteringAcc', train_clustering_info['QueryClusteringAcc'])
        summary.log(iteration, 'train/_TimeLoad', train_load_timer.interval)
        summary.log(iteration, 'train/_TimeBackprop', train_backprop_timer.interval)
        summary.log(iteration, 'train/TotalLoss', total_loss.item())  # Supervised accuracy

        # Sample from validation
        if iteration % 10 == 0 and val_iter is not None:
            with Timer() as val_load_timer:

                sample_val, __ = val_iter.next()

            with Timer() as val_eval_timer:

                # z = h(x)
                embedding_val = model.embed(sample_val, raw_input=opt['rawinput'])

                __, val_supervised_info = model.supervised_loss(embedding_val, regularization=regularization, supervised_sinkhorn_loss=opt['supervisedsinkhorn'])
                __, val_clustering_info = model.clustering_loss(embedding_val, regularization=regularization, supervised_sinkhorn_loss=opt['supervisedsinkhorn'])

            print 'Immediate', val_clustering_info['ClusteringAcc']

            summary.log(iteration, 'val/SupervisedAcc', val_supervised_info['SupervisedAccuracy'])
            summary.log(iteration, 'val/SupervisedLoss', val_supervised_info['SupervisedLoss'])
            summary.log(iteration, 'val/SupportClusteringAcc', val_clustering_info['SupportClusteringAcc'])
            summary.log(iteration, 'val/QueryClusteringAcc', val_clustering_info['QueryClusteringAcc'])
            summary.log(iteration, 'val/_TimeLoad', val_load_timer.interval)
            summary.log(iteration, 'val/_TimeEval', val_eval_timer.interval)

        # End of epoch? -> schedule new learning rate
        if new_epoch and iteration>0:
            print 'End of epoch, scheduling new learning rate'
            scheduler.step()

            summary.log(iteration, 'other/_LR', scheduler.get_lr())

        # Save model
        if iteration>0 and new_epoch:
            if opt['rawinput']:
                print 'No model to save in raw_input mode'
            else:
                print 'Saving model'
                model.cpu()
                torch.save(model, os.path.join(opt['log.exp_dir'], 'current_model.pt'))
                if opt['data.cuda']:
                    model.cuda()

        # Log

        if iteration % 10 == 0:
            print 'Iteration', iteration
            summary.print_summary()

        #### Save log
        if iteration % 500 == 0 or iteration < 10:
            try:
                with open(os.path.join(opt['log.exp_dir'], 'log.json'), 'wb') as fp:
                    json.dump(summary.logs, fp)
            except Exception as e:
                print 'Could not dump log file! Ignoring for now', e
