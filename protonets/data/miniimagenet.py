# Taken and modified from
# https://github.com/cyvius96/prototypical-network-pytorch

import os.path as osp
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torch.utils.data import DataLoader

from protonets.data.base import CudaTransform


class SimpleCudaTransform(object):
    def __call__(self, data):
        data = data.cuda()
        return data


class MiniImageNet(Dataset):

    def __init__(self, ROOT_PATH, setname, cuda=False):
        self.cuda = cuda

        csv_path = osp.join(ROOT_PATH, setname + '.csv')
        lines = [x.strip() for x in open(csv_path, 'r').readlines()][1:]

        data = []
        label = []
        lb = -1

        self.wnids = []

        for l in lines:
            name, wnid = l.split(',')
            path = osp.join(ROOT_PATH, 'images', name)
            if wnid not in self.wnids:
                self.wnids.append(wnid)
                lb += 1
            data.append(path)
            label.append(lb)

        self.data = data
        self.label = label

        t = [
            transforms.Resize(84),
            transforms.CenterCrop(84),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ]

        #if cuda:
        #    t.append(SimpleCudaTransform())

        self.transform = transforms.Compose(t)


    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        path, label = self.data[i], self.label[i]
        image = self.transform(Image.open(path).convert('RGB'))
        return image, label


class CategoriesSampler(object):

    def __init__(self, label, n_batch, n_cls, n_per):
        self.n_batch = n_batch
        self.n_cls = n_cls
        self.n_per = n_per

        label = np.array(label)
        self.m_ind = []
        for i in range(max(label) + 1):
            ind = np.argwhere(label == i).reshape(-1)
            ind = torch.from_numpy(ind)
            self.m_ind.append(ind)

    def __len__(self):
        return self.n_batch

    def __iter__(self):
        for i_batch in range(self.n_batch):
            batch = []
            classes = torch.randperm(len(self.m_ind))[:self.n_cls]
            for c in classes:
                l = self.m_ind[c]
                pos = torch.randperm(len(l))[:self.n_per]
                batch.append(l[pos])
            batch = torch.stack(batch).t().reshape(-1)
            yield batch


def AdapterDataLoader(DataLoader):

    def __init__(self, n_class, n_shot, n_query, *args, **kwargs):
        self.n_class = n_class
        self.n_shot = n_shot
        self.n_query = n_query
        DataLoader.__init__(self, *args, **kwargs)

    def __iter__(self):
        for batch in DataLoader:
            xs = batch[:self.shot * self.train_way]
            xq = batch[self.shot * self.train_way]
            sample = {

            }


def load(opt, splits):

    ret = { }
    for split in splits:
        if split in ['val', 'test'] and opt['data.test_way'] != 0:
            n_way = opt['data.test_way']
        else:
            n_way = opt['data.way']

        if split in ['val', 'test'] and opt['data.test_shot'] != 0:
            n_support = opt['data.test_shot']
        else:
            n_support = opt['data.shot']

        if split in ['val', 'test'] and opt['data.test_query'] != 0:
            n_query = opt['data.test_query']
        else:
            n_query = opt['data.query']

        if split in ['val', 'test']:
            n_episodes = opt['data.test_episodes']
        else:
            n_episodes = opt['data.train_episodes']


        # Right now CUDA is ignored. It will be used in the data adapter.
        # This is due to issues with multiprocessing and CUDA driver initialization.
        # https://github.com/pytorch/pytorch/issues/2517

        if split == 'train':
            trainset = MiniImageNet(opt['data.root'], 'train', cuda=opt['data.cuda'])
            train_sampler = CategoriesSampler(trainset.label, 100,
                                              n_way, n_support+n_query)
            train_loader = DataLoader(dataset=trainset, batch_sampler=train_sampler,
                                      num_workers=8, pin_memory=True)

            ret[split] = train_loader
        elif split == 'val':
            valset = MiniImageNet(opt['data.root'], 'val', cuda=opt['data.cuda'])
            val_sampler = CategoriesSampler(valset.label, 400,
                                            n_way, n_support+n_query)
            val_loader = DataLoader(dataset=valset, batch_sampler=val_sampler,
                                    num_workers=8, pin_memory=True)

            ret[split] = val_loader
        else:
            raise Exception('Split not implemented for MiniImagenet')


    return ret
