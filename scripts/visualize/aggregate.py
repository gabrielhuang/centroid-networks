import os
import numpy as np
import json
import scipy.ndimage.filters
import string
valid_chars = "-_.() %s%s" % (string.ascii_letters, string.digits)

import argparse

parser = argparse.ArgumentParser('')
parser.add_argument('--folder', default='../train/few_shot/', help='folder with all the runs')
parser.add_argument('--fraction', default=0.2, type=float, help='size of tail to use')
parser.add_argument('--skipfraction', default=0, type=float, help='fraction to skip in the end')
parser.add_argument('--filter', default='', help='filter runs by filename')
parser.add_argument('--gui', default=True, type=int, help='use GUI')
parser.add_argument('--version', default=3, type=int, help='which version')

args = parser.parse_args()


import matplotlib
if not args.gui:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt

filter = args.filter

#folder = '../../results_cluster'
folder = args.folder
#folder = '../train/few_shot/good/'
fraction = args.fraction
skipfraction = args.skipfraction

files = 0

stats = {}

for f in os.listdir(folder):
    path = os.path.join(folder, f)
    if os.path.isdir(path) and filter in path:
        print 'Reading', path
        try:
            with open(os.path.join(path, 'log.json'), 'rb') as fp:
                stats[f] = json.load(fp)
                print stats[f].keys()
        except Exception:
            print 'Skipping, file is incomplete'


# Useful keys
useful_keys_v1 = [
    'train/SupervisedAcc',
    'train/CentroidLoss',    'train/CentroidLossUnscaled',
    'val/SupervisedAcc',
    'val/SupportClusteringAcc',
    'val/QueryClusteringAcc',
    'other/_LR'
]


# Useful keys
useful_keys_v2 = [
    #'train/SupervisedAcc_softmax',
    'train/SupervisedAcc_sinkhorn',
    #'val/SupervisedAcc_softmax',
    'val/SupervisedAcc_sinkhorn',
    #'val/SupportClusteringAcc_softmax',
    'val/SupportClusteringAcc_sinkhorn',
    #'val/QueryClusteringAcc_softmax',
    'val/QueryClusteringAcc_sinkhorn',
    'other/_LR'
]

reg = 1.
useful_keys_v3 = [
    'train/CentroidLoss',
    'train/CentroidLossUnscaled',
    #'train/SupervisedAcc_softmax',
    'train/SupervisedAcc_sinkhorn',
    #'val/SupervisedAcc_softmax',
    'val/SupervisedAcc_sinkhorn',
    #'val/SupportClusteringAcc_softmax_reg{}'.format(reg),
    'val/SupportClusteringAcc_sinkhorn_reg{}'.format(reg),
    #'val/QueryClusteringAcc_softmax_reg{}'.format(reg),
    'val/QueryClusteringAcc_sinkhorn_reg{}'.format(reg),
    'other/_LR'
]

useful_keys = eval('useful_keys_v{}'.format(args.version))

#print stats
print 'Total of {} runs'.format(len(stats))


for key in useful_keys:
    plt.figure()
    print '*'*32
    print key
    print '*'*32
    # For all runs
    for f, stat in stats.items():

        x, y = zip(*sorted([(int(k), v) for k, v in stat[key].items()]))
        smoothed_y = scipy.ndimage.filters.gaussian_filter1d(y, sigma=31)

        print '\n{}  [{}]'.format(f, key)
        if skipfraction > 0 and skipfraction < fraction:
            subset_y = y[-int(len(y)*fraction):-int(len(y)*(skipfraction))]
            subset_y_smooth = smoothed_y[-int(len(y)*fraction):-int(len(y)*(skipfraction))]
            subset_x = x[-int(len(y)*fraction):-int(len(y)*(skipfraction))]
        else:
            subset_y = y[-int(len(y)*fraction):]
            subset_y_smooth = smoothed_y[-int(len(y)*fraction):]
            subset_x = x[-int(len(y)*fraction):]
        print '\tAverage over last {} samples:'.format(len(subset_y))
        print '\titerations {} -> {}'.format(subset_x[0], subset_x[-1])
        print '\t{:.4f} +/- {:.4f} (std = {:.4f})'.format(np.mean(subset_y), np.std(subset_y)/float(np.sqrt(len(subset_y))), np.std(subset_y))

        plt.plot(subset_x, subset_y_smooth, label=f)

    plt.legend()
    plt.title(key)

    valid_key = ''.join(c for c in key if c in valid_chars)
    plt.savefig('plot_{}.png'.format(valid_key))



plt.show()

