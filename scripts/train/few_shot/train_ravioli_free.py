import os
import json
from functools import partial
from tqdm import tqdm

import numpy as np

import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torchvision
import torchnet as tnt

from protonets.engine import Engine

import protonets.utils.data as data_utils
import protonets.utils.model as model_utils
import protonets.utils.log as log_utils

def main(opt):
    ###########################################
    # Boilerplate
    ###########################################
    if not os.path.isdir(opt['log.exp_dir']):
        os.makedirs(opt['log.exp_dir'])

    # save opts
    with open(os.path.join(opt['log.exp_dir'], 'opt.json'), 'w') as f:
        json.dump(opt, f)
        f.write('\n')

    trace_file = os.path.join(opt['log.exp_dir'], 'trace.txt')

    # Postprocess arguments
    opt['model.x_dim'] = list(map(int, opt['model.x_dim'].split(',')))
    opt['log.fields'] = opt['log.fields'].split(',')

    torch.manual_seed(1234)
    if opt['data.cuda']:
        torch.cuda.manual_seed(1234)

    if opt['data.trainval']:
        data = data_utils.load(opt, ['trainval'])
        train_loader = data['trainval']
        val_loader = None
    else:
        data = data_utils.load(opt, ['train', 'val'])
        train_loader = data['train']
        val_loader = data['val']

    ###########################################
    # Create model
    ###########################################
    model = model_utils.load(opt)

    if opt['data.cuda']:
        model.cuda()


    meters = { 'train': { field: tnt.meter.AverageValueMeter() for field in opt['log.fields'] } }

    if val_loader is not None:
        meters['val'] = { field: tnt.meter.AverageValueMeter() for field in opt['log.fields'] }


    ###########################################
    # Training loop
    ###########################################

    #### Start of training loop
    state = {
        'model': model,
        'loader': train_loader,
        'optim_method': getattr(optim, opt['train.optim_method']),
        'optim_config': { 'lr': opt['train.learning_rate'],
                         'weight_decay': opt['train.weight_decay'] },
        'max_epoch': opt['train.epochs'],
        'epoch': 0,  # epochs done so far
        't': 0,  # samples seen so far
        'batch': 0,  # samples seen in current epoch
        'stop': False
    }

    state['optimizer'] = state['optim_method'](state['model'].parameters(), **state['optim_config'])

    hook_state = {}  # this was passed as { } when registering hook


    ###### Start hook
    if os.path.isfile(trace_file):
        os.remove(trace_file)
    state['scheduler'] = lr_scheduler.StepLR(state['optimizer'], opt['train.decay_every'], gamma=0.5)
    ######

    while state['epoch'] < state['max_epoch'] and not state['stop']:
        state['model'].train()

        ###### Start Epoch hook
        for split, split_meters in meters.items():
            for field, meter in split_meters.items():
                meter.reset()
        state['scheduler'].step()
        ######

        state['epoch_size'] = len(state['loader'])

        for sample in tqdm(state['loader'], desc="Epoch {:d} train".format(state['epoch'] + 1)):
            state['sample'] = sample

            state['optimizer'].zero_grad()
            loss, state['output'] = state['model'].loss(state['sample'])

            loss.backward()

            state['optimizer'].step()

            state['t'] += 1
            state['batch'] += 1

            #### On update
            for field, meter in meters['train'].items():
                meter.add(state['output'][field])
            if state['batch'] % 5 == 0:
                meter_vals = log_utils.extract_meter_values(meters)
                print("Epoch {:02d}: {:s}".format(state['epoch'], log_utils.render_meter_values(meter_vals)))
            ####

            break

        state['epoch'] += 1
        state['batch'] = 0

        #### On end epoch
        if val_loader is not None:
            if 'best_loss' not in hook_state:
                hook_state['best_loss'] = np.inf
            if 'wait' not in hook_state:
                hook_state['wait'] = 0

        if val_loader is not None:
            #### Model evaluate
            model.eval()

            for field, meter in meters['val'].items():
                meter.reset()

            data_loader = tqdm(val_loader, desc="Epoch {:d} valid".format(state['epoch']))

            for sample in val_loader:
                _, output = model.eval_loss(sample)
                for field, meter in meters['val'].items():
                    meter.add(output[field])

                break

        meter_vals = log_utils.extract_meter_values(meters)
        print("Epoch {:02d}: {:s}".format(state['epoch'], log_utils.render_meter_values(meter_vals)))
        meter_vals['epoch'] = state['epoch']
        with open(trace_file, 'a') as f:
            json.dump(meter_vals, f)
            f.write('\n')

        if val_loader is not None:
            if meter_vals['val']['loss'] < hook_state['best_loss']:
                hook_state['best_loss'] = meter_vals['val']['loss']
                print("==> best model (loss = {:0.6f}), saving model...".format(hook_state['best_loss']))

                state['model'].cpu()
                torch.save(state['model'], os.path.join(opt['log.exp_dir'], 'best_model.pt'))
                if opt['data.cuda']:
                    state['model'].cuda()

                hook_state['wait'] = 0
            else:
                hook_state['wait'] += 1

                if hook_state['wait'] > opt['train.patience']:
                    print("==> patience {:d} exceeded".format(opt['train.patience']))
                    state['stop'] = True
        else:
            state['model'].cpu()
            torch.save(state['model'], os.path.join(opt['log.exp_dir'], 'best_model.pt'))
            if opt['data.cuda']:
                state['model'].cuda()
