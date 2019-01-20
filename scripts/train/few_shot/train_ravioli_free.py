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


def data_adapter(iterator, opt, train):
    '''
    Adapter for miniimagenet loader (taken from other code)
    '''
    assert opt['data.way'] == opt['data.test_way']
    for (sample, new_epoch) in iterator:
        if opt['data.dataset'] == 'miniimagenet':
            (x, y) = sample

            if train:
                n_way = opt['data.way']
                n_shot = opt['data.shot']
                n_query = opt['data.query']
            else:
                n_way = opt['data.test_way']
                n_shot = opt['data.test_shot'] or opt['data.shot']
                n_query = opt['data.test_query'] or opt['data.query']

            # TODO: check this is fine -> it was not fine
            x = x.view(n_way, n_shot + n_query, *x.size()[1:])
            xs = x[:, :n_shot].contiguous()
            xq = x[:, n_shot:].contiguous()

            if opt['data.cuda']:
                xs = xs.cuda()
                xq = xq.cuda()

            yield {
                'xs': xs,
                'xq': xq,
                'class': 'No class for now'
            }, new_epoch
        elif opt['data.dataset'] == 'omniglot':
            yield sample, new_epoch
        else:
            raise Exception('Unregistered dataset')

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

    # Adapt model size to dataset
    if opt['data.dataset'] == 'omniglot':
        opt['model.x_dim'] = '1,28,28'
    elif opt['data.dataset'] == 'miniimagenet':
        opt['model.x_dim'] = '3,84,84'


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
        train_iter = data_adapter(make_infinite(train_loader), opt, train=True)
        val_iter = None
    else:
        data = data_utils.load(opt, ['train', 'val'])
        train_loader = data['train']
        val_loader = data['val']

        # Prepare datasets
        train_iter = data_adapter(make_infinite(train_loader), opt, train=True)
        val_iter = data_adapter(make_infinite(val_loader), opt, train=False)

    ###########################################
    # Create model and optimizer
    ###########################################

    model = model_utils.load(opt)

    if opt['checkpoint']:
        print 'Loading from checkpoint', opt['checkpoint']
        model = torch.load(opt['checkpoint'])

    if opt['checkpoint_state']:
        print 'Loading state from checkpoint', opt['checkpoint_state']
        model.load_state_dict(torch.load(opt['checkpoint_state'], map_location=lambda storage, loc: storage))

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
            # For debug
            #plt.imshow(0.5 + 0.5 * np.rollaxis(sample_train['xs'].numpy(), 2, 5)[0].reshape((5 * 84, 84, 3)))

        # Compute loss; backprop
        with Timer() as train_backprop_timer:

            model.train()  # batchnorm train mode

            # z = h(x)
            embedding_train = model.embed(sample_train, raw_input=opt['rawinput'])

            if iteration == 0:
                print 'Debug: Tensor sizes'
                print 'xs', sample_train['xs'].size()
                print 'xq', sample_train['xq'].size()
                print 'zs', embedding_train['zs'].size()
                print 'zq', embedding_train['zq'].size()
                # Should be 64 for omniglot and 1600 for miniimagenet

            # Supervised and Clustering Losses
            train_supervised_info = model.supervised_loss(embedding_train, regularization=regularization)
            train_clustering_info = model.clustering_loss(embedding_train, regularization=regularization)

            if opt['train_loss'] == 'softmax':  # softmax
                total_loss = train_supervised_info['SupervisedLoss_softmax']
            elif opt['train_loss'] == 'sinkhorn':
                total_loss = train_supervised_info['SupervisedLoss_sinkhorn']
            elif opt['train_loss'] == 'evalonly':
                total_loss = torch.zeros([])
            else:
                raise Exception('Unknown meta-training loss {}'.format(opt['train_loss']))

            if opt['centroid_loss'] > 0. :
                centroid_loss = opt['centroid_loss'] * train_supervised_info['ClassVariance']
                total_loss = total_loss + centroid_loss
                summary.log(iteration, 'train/CentroidLoss', centroid_loss.item())  # Supervised accuracy
            summary.log(iteration, 'train/CentroidLossUnscaled', train_supervised_info['ClassVariance'].item())  # Supervised accuracy

            if not opt['rawinput'] and opt['train_loss'] != 'evalonly':
                # No need to backprop in rawinput mode
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

        # supervised losses
        summary.log(iteration, 'train/SupervisedAcc_softmax', train_supervised_info['SupervisedAcc_softmax'].item())
        summary.log(iteration, 'train/SupervisedAcc_sinkhorn', train_supervised_info['SupervisedAcc_sinkhorn'].item())
        summary.log(iteration, 'train/SupervisedLoss_softmax', train_supervised_info['SupervisedLoss_softmax'].item())
        summary.log(iteration, 'train/SupervisedLoss_sinkhorn', train_supervised_info['SupervisedLoss_sinkhorn'].item())

        # unsupervised losses
        summary.log(iteration, 'train/SupportClusteringAcc_softmax', train_clustering_info['SupportClusteringAcc_softmax'])
        summary.log(iteration, 'train/SupportClusteringAcc_sinkhorn', train_clustering_info['SupportClusteringAcc_sinkhorn'])
        summary.log(iteration, 'train/QueryClusteringAcc_softmax', train_clustering_info['QueryClusteringAcc_softmax'])
        summary.log(iteration, 'train/QueryClusteringAcc_sinkhorn', train_clustering_info['QueryClusteringAcc_sinkhorn'])

        summary.log(iteration, 'train/_TimeLoad', train_load_timer.interval)
        summary.log(iteration, 'train/_TimeBackprop', train_backprop_timer.interval)
        summary.log(iteration, 'train/TotalLoss', total_loss.item())  # Supervised accuracy

        # Sample from validation
        if iteration % 10 == 0 and val_iter is not None:
            with Timer() as val_load_timer:

                sample_val, __ = val_iter.next()

            with Timer() as val_eval_timer:

                # Weird? deactivate batchnorm train mode
                model.eval()

                # z = h(x)
                embedding_val = model.embed(sample_val, raw_input=opt['rawinput'])

                val_supervised_info = model.supervised_loss(embedding_val, regularization=regularization)
                val_clustering_info = model.clustering_loss(embedding_val, regularization=regularization)

            print 'Immediate', val_clustering_info['SupportClusteringAcc_sinkhorn']

            # supervised losses
            summary.log(iteration, 'val/SupervisedAcc_softmax', val_supervised_info['SupervisedAcc_softmax'].item())
            summary.log(iteration, 'val/SupervisedAcc_sinkhorn', val_supervised_info['SupervisedAcc_sinkhorn'].item())
            summary.log(iteration, 'val/SupervisedLoss_softmax', val_supervised_info['SupervisedLoss_softmax'].item())
            summary.log(iteration, 'val/SupervisedLoss_sinkhorn', val_supervised_info['SupervisedLoss_sinkhorn'].item())

            # unsupervised losses
            summary.log(iteration, 'val/SupportClusteringAcc_softmax', val_clustering_info['SupportClusteringAcc_softmax'])
            summary.log(iteration, 'val/SupportClusteringAcc_sinkhorn', val_clustering_info['SupportClusteringAcc_sinkhorn'])
            summary.log(iteration, 'val/QueryClusteringAcc_softmax', val_clustering_info['QueryClusteringAcc_softmax'])
            summary.log(iteration, 'val/QueryClusteringAcc_sinkhorn', val_clustering_info['QueryClusteringAcc_sinkhorn'])

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
