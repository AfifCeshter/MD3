#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = "Han"
__email__ = "liuhan132@foxmail.com"

import logging
import torch
import torch.utils.data
from game.template import AgentActs
from .base_reader import BaseReader

logger = logging.getLogger(__name__)


def load_docs_rep(data_path_prefix, on_cpu=True):
    """
    Load documents representation by pytorch binary file
    :param on_cpu:
    :param data_path_prefix:
    :return:
    """
    docs_rep = []

    for slot in AgentActs.ALL_SLOTS:
        if on_cpu:
            docs_rep_slot = torch.load(data_path_prefix + '-' + slot, map_location=torch.device('cpu'))
        else:
            docs_rep_slot = torch.load(data_path_prefix + '-' + slot)
        docs_rep.append(docs_rep_slot)

    # (num_docs, num_slots, hidden_size * 4)
    docs_rep = torch.stack(docs_rep, dim=1)
    return docs_rep


def get_docs_diff_rep(docs_rep):
    docs_mean_rep = docs_rep.mean(dim=0)    # (num_slots, hidden_size * 4)
    docs_diff_rep = (docs_rep - docs_mean_rep.unsqueeze(0)).pow(2)

    return docs_diff_rep


class DocRepPTReader(BaseReader):
    """
    Documents representation reader(For pre-processing)
    """

    def __init__(self, game_config):
        super(DocRepPTReader, self).__init__(game_config)

    def get_dataloader_docs(self, batch_size, num_workers):
        """
        Documents dataloader for documents representation
        :param batch_size:
        :param num_workers:
        :param shuffle:
        :return:
        """
        doc_dataset = DocRepPTDataset(self.doc_reader.get_all_names(),
                                      self.doc_to_tensor)
        return torch.utils.data.DataLoader(doc_dataset,
                                           batch_size=batch_size,
                                           num_workers=num_workers,
                                           shuffle=False)


class DocRepPTDataset(torch.utils.data.Dataset):
    """
    Documents representation dataset
    """

    def __init__(self, doc_names, doc_to_tensor):
        super(DocRepPTDataset, self).__init__()

        self.doc_to_tensor = doc_to_tensor
        self.doc_names = doc_names

    def __len__(self):
        return len(self.doc_names)

    def __getitem__(self, index):
        name = self.doc_names[index]
        doc_tensor = self.doc_to_tensor([name], hierarchical=True)[0]

        return doc_tensor
