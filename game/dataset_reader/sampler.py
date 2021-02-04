#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = "Han"
__email__ = "liuhan132@foxmail.com"


import random
import numpy as np
import torch
from torch.utils.data.sampler import Sampler


class ValidDocBatchSampler(Sampler):

    def __init__(self, valid_doc_nums, batch_size, shuffle=True, strict=True):
        super(ValidDocBatchSampler, self).__init__(valid_doc_nums)

        self.valid_doc_nums = valid_doc_nums
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.strict = strict

    def __iter__(self):
        lengths = np.array(
            [(-l, np.random.random()) for l in self.valid_doc_nums],
            dtype=[('l1', np.int_), ('rand', np.float_)]
        )
        indices = np.argsort(lengths, order=('l1', 'rand'))
        batches = [indices[i:i + self.batch_size]
                   for i in range(0, len(indices), self.batch_size)]

        if self.strict:
            batches = list(filter(
                lambda x: sum([self.valid_doc_nums[i] for i in x]) == self.valid_doc_nums[x[0]] * len(x),
                batches
            ))

        last = batches[-1]

        if self.shuffle:
            batches = batches[:len(batches) - 1]
            np.random.shuffle(batches)
            batches.append(last)
        return iter([i for batch in batches for i in batch])

    def __len__(self):
        return len(self.valid_doc_nums)


class DialogTurnsSampler(Sampler):

    def __init__(self, turns_num, batch_size, iter_num):
        super(DialogTurnsSampler, self).__init__(turns_num)

        self.turns_num = turns_num
        self.batch_size = batch_size
        self.iter_num = iter_num

    def __iter__(self):
        lengths = np.array(
            [(-l, np.random.random()) for l in self.turns_num],
            dtype=[('l1', np.int_), ('rand', np.float_)]
        )
        indices = np.argsort(lengths, order=('l1', 'rand'))
        batches = [indices[i:i + self.batch_size]
                   for i in range(0, len(indices), self.batch_size)]

        batches_i = iter(torch.randint(high=len(batches), size=(self.iter_num,), dtype=torch.int64))
        return iter([i for batch_i in batches_i for i in batches[batch_i]])

    def __len__(self):
        return self.iter_num * self.batch_size


class GenBatchSampleIter:

    def __init__(self, dataset, batch_size, collect_fun):
        self.batch_size = batch_size
        self.dataset = dataset
        self.collect_fun = collect_fun

    def __len__(self):
        return len(self.dataset)

    def __next__(self):
        batch_idx = random.sample(range(len(self.dataset)), self.batch_size)
        batch_data = [self.dataset[i] for i in batch_idx]

        return self.collect_fun(batch_data)
