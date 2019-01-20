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
#useful_keys = ['val/acc']
useful_keys = ['val/QueryClusteringAcc', 'val/SupportClusteringAcc', 'train/SupervisedAcc', 'val/SupervisedAcc']

#print stats
print 'Total of {} runs'.format(len(stats))


for key in useful_keys:
    plt.figure()
    # For all runs
    for f, stat in stats.items():
        x, y = zip(*sorted([(int(k), v) for k, v in stat[key].items()]))
        smoothed_y = scipy.ndimage.filters.gaussian_filter1d(y, sigma=101)
        plt.plot(x, smoothed_y, label=f)

        print '\n{}  [{}]'.format(f, key)
        if skipfraction > 0 and skipfraction < fraction:
            subset_y = y[-int(len(y)*fraction):-int(len(y)*(fraction-skipfraction))]
            subset_x = x[-int(len(y)*fraction):-int(len(y)*(fraction-skipfraction))]
        else:
            subset_y = y[-int(len(y)*fraction):]
            subset_x = x[-int(len(y)*fraction):]
        print '\tAverage over last {} samples:'.format(len(subset_y))
        print '\titerations {} -> {}'.format(subset_x[0], subset_x[-1])
        print '\t{:.4f} +/- {:.4f} (std = {:.4f})'.format(np.mean(subset_y), np.std(subset_y)/float(np.sqrt(len(subset_y))), np.std(subset_y))
    plt.legend()
    plt.title(key)

    valid_key = ''.join(c for c in key if c in valid_chars)
    plt.savefig('plot_{}.png'.format(valid_key))



plt.show()

