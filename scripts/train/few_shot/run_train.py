import argparse

import train
import train_ravioli_free

parser = argparse.ArgumentParser(description='Train prototypical networks')

# data args
default_dataset = 'omniglot'
parser.add_argument('--data.dataset', type=str, default=default_dataset, choices=['omniglot','omniglot_ccn', 'miniimagenet'],
                    metavar='DS', help="data set name (default: {:s})".format(default_dataset))
parser.add_argument('--data.root', default='', help='path to dataset')
default_split = 'vinyals'
parser.add_argument('--data.split', type=str, default=default_split, metavar='SP',
                    help="split name (default: {:s})".format(default_split))
parser.add_argument('--data.way', type=int, default=5, metavar='WAY',
                    help="number of classes per episode (default: 60)")
parser.add_argument('--data.shot', type=int, default=5, metavar='SHOT',
                    help="number of support examples per class (default: 5)")
parser.add_argument('--data.query', type=int, default=5, metavar='QUERY',
                    help="number of query examples per class (default: 5)")
parser.add_argument('--data.test_way', type=int, default=5, metavar='TESTWAY',
                    help="number of classes per episode in test. 0 means same as data.way (default: 5)")
parser.add_argument('--data.test_shot', type=int, default=0, metavar='TESTSHOT',
                    help="number of support examples per class in test. 0 means same as data.shot (default: 0)")
parser.add_argument('--data.test_query', type=int, default=15, metavar='TESTQUERY',
                    help="number of query examples per class in test. 0 means same as data.query (default: 15)")
parser.add_argument('--data.train_episodes', type=int, default=100, metavar='NTRAIN',
                    help="number of train episodes per epoch (default: 100)")
parser.add_argument('--data.test_episodes', type=int, default=100, metavar='NTEST',
                    help="number of test episodes per epoch (default: 100)")
parser.add_argument('--data.trainval', action='store_true', help="run in train+validation mode (default: False)")
parser.add_argument('--data.sequential', action='store_true', help="use sequential sampler instead of episodic (default: False)")
parser.add_argument('--data.cuda', action='store_true', help="run in CUDA mode (default: False)")

# model args
default_model_name = 'protonet_conv'
parser.add_argument('--model.model_name', type=str, default=default_model_name, metavar='MODELNAME',
                    help="model name (default: {:s})".format(default_model_name))
parser.add_argument('--model.x_dim', type=str, default='1,28,28', metavar='XDIM',
                    help="dimensionality of input images (default: '1,28,28')")
parser.add_argument('--model.hid_dim', type=int, default=64, metavar='HIDDIM',
                    help="dimensionality of hidden layers (default: 64)")
parser.add_argument('--model.z_dim', type=int, default=64, metavar='ZDIM',
                    help="dimensionality of input images (default: 64)")

# train args
parser.add_argument('--train.epochs', type=int, default=10000, metavar='NEPOCHS',
                    help='number of epochs to train (default: 10000)')
parser.add_argument('--train.optim_method', type=str, default='Adam', metavar='OPTIM',
                    help='optimization method (default: Adam)')
parser.add_argument('--train.learning_rate', type=float, default=0.001, metavar='LR',
                    help='learning rate (default: 0.0001)')
parser.add_argument('--train.decay_every', type=int, default=20, metavar='LRDECAY',
                    help='number of epochs after which to decay the learning rate')
default_weight_decay = 0.0
parser.add_argument('--train.weight_decay', type=float, default=default_weight_decay, metavar='WD',
                    help="weight decay (default: {:f})".format(default_weight_decay))
parser.add_argument('--train.patience', type=int, default=200, metavar='PATIENCE',
                    help='number of epochs to wait before validation improvement (default: 1000)')

# log args
default_fields = 'loss,acc'
parser.add_argument('--log.fields', type=str, default=default_fields, metavar='FIELDS',
                    help="fields to monitor during training (default: {:s})".format(default_fields))
default_exp_dir = 'results'
parser.add_argument('--log.exp_dir', type=str, default=default_exp_dir, metavar='EXP_DIR',
                    help="directory where experiments should be saved (default: {:s})".format(default_exp_dir))

# which version to use
parser.add_argument('--ravioli', type=int, default=0, help='1: use original code; 0: big fat script')

# Unsupervised Few-shot learning specific parameters
parser.add_argument('--iterations', default=100000, type=int, help='number of iterations (i.e. episodes)')
parser.add_argument('--validate-interval', default=10, type=int, help='validate every X training iterations')
parser.add_argument('--checkpoint', default='', help='load model checkpoint')
parser.add_argument('--checkpoint-state', default='', help='load model state checkpoint')
parser.add_argument('--centroid-loss', default=0., type=float, help='centroid loss penalty')
parser.add_argument('--train-loss', required=True, choices=['softmax', 'sinkhorn', 'twostep', 'evalonly'], help='meta-training loss')
parser.add_argument('--temperature', default=1., type=float, help='temperature for softmax and assignments')
parser.add_argument('--regularizations', default='0.01,0.1,10,100,1', type=str, help='regularizations to try')  # 1 should be the last one
parser.add_argument('--rawinput', default=0, type=int, help='use raw inputs or train features (model weights are ignored)')
parser.add_argument('--hide-test', default=1, type=int, help='hide meta-testing metrics (they will be logged)')
parser.add_argument('--clustering', default='wasserstein', choices=['kmeans', 'kmeansplusplus', 'wasserstein'], help='which clustering algorithm to use')
parser.add_argument('--sanity-check', default=0, type=int, help='Sanity check: permute labels and data before clustering')

args = vars(parser.parse_args())

if args['train_loss'] == 'evalonly':
    args['iteration'] = 1000
    args['validate_interval'] = 1
    args['hide_test'] = 0

    # Append .eval to dir name
    model_dir = args['log.exp_dir']
    args['log.exp_dir'] = '{}.eval'.format(model_dir)

    # Take best model in folder
    if not args['checkpoint'] and not args['checkpoint_state'] and not args['rawinput']:
        args['checkpoint'] = '{}/current_model.pt'.format(model_dir)

    assert (args['checkpoint'] or args['checkpoint_state']) or args['rawinput'], 'Really? Evaluate untrained model?'

if args['ravioli']:
    train.main(args)
else:
    train_ravioli_free.main(args)
