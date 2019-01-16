import os
import numpy as np
import json
import matplotlib.pyplot as plt
import scipy.ndimage.filters

filter = '20way'

#folder = '../../results_cluster'
folder = '../train/few_shot/'
#folder = '../train/few_shot/good/'
fraction = 0.2

files = 0

stats = {}

for f in os.listdir(folder):
    path = os.path.join(folder, f)
    if os.path.isdir(path) and 'results.' in path and filter in path:
        print 'Reading', path
        try:
            with open(os.path.join(path, 'log.json'), 'rb') as fp:
                stats[f] = json.load(fp)
                print stats[f].keys()
        except Exception:
            print 'Skipping, file is incomplete'

# Useful keys
#useful_keys = ['val/acc']
useful_keys = ['val/ClusteringAcc', 'train/ClusteringAcc', 'train/SupervisedAcc', 'train/CentroidLossUnscaled']

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
        subset_y = y[-int(len(y)*fraction):]
        subset_x = x[-int(len(y)*fraction):]
        print '\tAverage over last {} samples:'.format(len(subset_y))
        print '\titerations {} -> {}'.format(subset_x[0], subset_x[-1])
        print '\t{:.4f} +/- {:.4f} (std = {:.4f})'.format(np.mean(subset_y), np.std(subset_y)/float(np.sqrt(len(subset_y))), np.std(subset_y))
    plt.legend()
    plt.title(key)



plt.show()
