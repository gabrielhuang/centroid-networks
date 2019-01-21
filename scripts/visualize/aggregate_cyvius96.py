import numpy as np
import torch
import matplotlib
import matplotlib.pyplot as plt

logs = []
logs.append('../../external/proto-5/trlog')
logs.append('../../external/proto-5-temperature1600/trlog')

skipfraction = 0
fraction = 0.1

for logname in logs:

    log = torch.load(logname)

    print log.keys()
    plt.figure()
    plt.plot(log['train_acc'], label='train')
    plt.plot(log['val_acc'], label='val')
    plt.legend()
    plt.title(logname)

    for key in ('train_acc', 'val_acc'):
        print '\n{}  [{}]'.format(logname, key)
        x = range(len(log[key]))
        y = log[key]
        if skipfraction > 0 and skipfraction < fraction:
            subset_y = y[-int(len(y) * fraction):-int(len(y) * (fraction - skipfraction))]
            subset_x = x[-int(len(y) * fraction):-int(len(y) * (fraction - skipfraction))]
        else:
            subset_y = y[-int(len(y) * fraction):]
            subset_x = x[-int(len(y) * fraction):]
        print '\tAverage over last {} samples:'.format(len(subset_y))
        print '\titerations {} -> {}'.format(subset_x[0], subset_x[-1])
        print '\t{:.4f} +/- {:.4f} (std = {:.4f})'.format(np.mean(subset_y),
                                                          np.std(subset_y) / float(np.sqrt(len(subset_y))),
                                                          np.std(subset_y))

plt.show()

# Average the last 10%

