# This file loads the omniglot dataset following
# the same scheme as Learning to cluster/CCN
# https://github.com/GT-RIPL/L2C

import os
import sys
import glob

from functools import partial

import numpy as np
from PIL import Image

import torch
from torchvision.transforms import ToTensor
import torchvision
from torchvision import transforms

from torchnet.dataset import ListDataset, TransformDataset
from torchnet.transform import compose

import protonets
from protonets.data.base import convert_dict, CudaTransform, EpisodicBatchSampler, SequentialBatchSampler


binary_flip = transforms.Lambda(lambda x: 1 - x)
normalize = transforms.Normalize((0.086,), (0.235,))

alphabet_images_cache = {}

class OmniglotCCNLoader(object):
    def __init__(self, split, n_batch=100, cuda=True):

        self.n_batch = 100 # not very important except for LR scheduling
        self.cuda = cuda

        # copied from L2C repo
        self.split = split
        background = (split == 'train')
        self.omniglot = torchvision.datasets.Omniglot(
            root='data', download=True, background=background,
            transform=transforms.Compose(
               [transforms.Resize(32),
                transforms.ToTensor(),
                binary_flip,
                normalize]
            )
        )

        # Get character indices for each alphabet (about 1623 in total)
        character_indices = {}
        for alphabet in self.omniglot._alphabets:
            character_indices[alphabet] = [i for i, character_name in enumerate(self.omniglot._characters) if alphabet in character_name]

        # Get instance indices for each character (about 1623*20 = 32460)
        instance_indices = {}
        for i, (__, y) in enumerate(self.omniglot._flat_character_images):
            instance_indices.setdefault(y, [])
            instance_indices[y].append(i)

        self.character_indices = character_indices
        self.instance_indices = instance_indices


    def get_alphabet_images(self, alphabet):

        if alphabet not in alphabet_images_cache:
            print 'Split {} -> Loading alphabet {} for the first time'.format(self.split, alphabet)
            # Return all its characters in a tensor
            alphabet_images = []
            for character in self.character_indices[alphabet]:
                assert len(self.instance_indices[character]) == 20
                # Concatenate all images of same character together
                character_images = []
                for instance in self.instance_indices[character]:
                    image, label = self.omniglot[instance]
                    character_images.append(image)
                # Concatenate all
                character_images = torch.stack(character_images)
                alphabet_images.append(character_images)
            alphabet_images = torch.stack(alphabet_images)
            alphabet_images_cache[alphabet] = alphabet_images

        return alphabet_images_cache[alphabet]

    def __iter__(self):

        for i in xrange(self.n_batch):
            # Sample an alphabet
            alphabet = self.omniglot._alphabets[np.random.randint(len(self.omniglot._alphabets))]
            # Get tensor of all characters
            alphabet_images = self.get_alphabet_images(alphabet)
            # Return
            if self.cuda:
                alphabet_images = alphabet_images.cuda()
            yield {
                'xs': alphabet_images,
                'xq': alphabet_images.clone(),  # preserve compatibility with centroid networks code
                'class': None,
                'alphabet': alphabet
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

        # Load custom loader
        omniglot_loader = OmniglotCCNLoader(split, n_batch=n_episodes, cuda=opt['data.cuda'])

        # use num_workers=0, otherwise may receive duplicate episodes
        ret[split] = omniglot_loader

    return ret

