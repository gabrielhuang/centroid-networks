import protonets.data

def load(opt, splits):
    if opt['data.dataset'] == 'omniglot':
        ds = protonets.data.omniglot.load(opt, splits)
    elif opt['data.dataset'] == 'miniimagenet':
        ds = protonets.data.miniimagenet.load(opt, splits)
    else:
        raise ValueError("Unknown dataset: {:s}".format(opt['data.dataset']))

    return ds
