#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = "Han"
__email__ = "liuhan132@foxmail.com"

import os
import logging
import json
import random
import torch
import torch.utils.data
from .base_reader import BaseReader
from .doc_dataset import DocDataset
from game.template import AgentActs
from utils.functions import detect_unk_kb, detect_same_kb

logger = logging.getLogger(__name__)


class DocPTReader(BaseReader):
    """
    Documents difference reader(For train)
    """

    def __init__(self, game_config):
        super(DocPTReader, self).__init__(game_config)

        self.cand_doc_num = game_config['global']['cand_doc_num']
        self.doc_hierarchical = game_config['model']['doc_hierarchical']

    def get_dataset_slot(self, slot, batch_size, num_workers, iters, data_path,
                         doc_rep_same_name_path, doc_rep_unk_name_path):
        """
        Dialog data loader for supervised training
        :param slot:
        :param iters:
        :param num_workers:
        :param doc_rep_unk_name_path:
        :param doc_rep_same_name_path:
        :param data_path:
        :param batch_size:
        :return:
        """
        doc_dataset = DocPTDataset(slot,
                                   self.doc_to_tensor,
                                   self.cand_doc_num,
                                   data_path=data_path,
                                   doc_rep_same_name_path=doc_rep_same_name_path,
                                   doc_rep_unk_name_path=doc_rep_unk_name_path,
                                   hierarchical=self.doc_hierarchical)
        cur_sampler = torch.utils.data.sampler.RandomSampler(doc_dataset,
                                                             replacement=True,
                                                             num_samples=iters * batch_size)
        return torch.utils.data.DataLoader(doc_dataset,
                                           batch_size=batch_size,
                                           sampler=cur_sampler,
                                           num_workers=num_workers,
                                           collate_fn=DocPTDataset.collect_fun)
        # return GenBatchSampleIter(doc_dataset, batch_size, DocPTDataset.collect_fun)

    def get_dataset_train_slot(self, slot, batch_size, num_workers, iters):
        return self.get_dataset_slot(slot, batch_size,
                                     num_workers=num_workers,
                                     iters=iters,
                                     data_path=self.data_prefix + 'doc_rep/guessmovie_rep_doc_id.json',
                                     doc_rep_same_name_path=self.data_prefix + 'doc_rep/rep_doc_same_name.json',
                                     doc_rep_unk_name_path=self.data_prefix + 'doc_rep/rep_doc_unk_name.json')

    def get_dataset_test_slot(self, slot, batch_size, num_workers, iters):
        return self.get_dataset_slot(slot, batch_size,
                                     num_workers=num_workers,
                                     iters=iters,
                                     data_path=self.data_prefix + 'guessmovie_dialog_doc_id.json',
                                     doc_rep_same_name_path=self.data_prefix + 'dialog_doc_same_name.json',
                                     doc_rep_unk_name_path=self.data_prefix + 'dialog_doc_unk_name.json')


class DocPTDataset(torch.utils.data.Dataset):
    """
    Documents difference dataset
    """

    def __init__(self, slot, doc_to_tensor, cand_doc_num,
                 data_path, doc_rep_same_name_path, doc_rep_unk_name_path, hierarchical):
        super(DocPTDataset, self).__init__()

        self.slot = slot
        self.cand_doc_num = cand_doc_num
        self.doc_to_tensor = doc_to_tensor
        self.hierarchical = hierarchical

        # build documents dataset reader
        self.doc_reader = DocDataset(data_path)
        self.all_docs_names = self.doc_reader.get_all_names()

        # same name data
        if not os.path.exists(doc_rep_same_name_path):
            detect_same_kb(data_path, doc_rep_same_name_path)

        with open(doc_rep_same_name_path, 'r') as f:
            self.doc_rep_same_name_data = json.load(f)

        # unk name data
        if not os.path.exists(doc_rep_unk_name_path):
            detect_unk_kb(data_path, doc_rep_unk_name_path)

        with open(doc_rep_unk_name_path, 'r') as f:
            self.doc_rep_unk_name_data = json.load(f)

    def __len__(self):
        return len(self.all_docs_names)

    def __getitem__(self, index):
        # self.slot = random.choice(AgentActs.ALL_SLOTS)      # TODO: random a slot

        # find the target document that not unk
        is_unk_slot = True
        while is_unk_slot:
            tar_doc_name = random.choice(self.all_docs_names)
            tar_doc_kb = self.doc_reader.get_kb(tar_doc_name, is_full=False)

            if self.slot in tar_doc_kb:
                is_unk_slot = False

        # select the document that have the same key and value
        tar_values = tar_doc_kb[self.slot]
        same_docs = set()
        for v in tar_values:
            same_docs.update(set(self.doc_rep_same_name_data[self.slot][v]))
        same_docs = list(same_docs)

        if len(same_docs) == 1:
            same_docs = self.doc_rep_unk_name_data[self.slot]

        # shouldn't be the same documents
        # same_docs.remove(tar_doc_name)
        assert len(same_docs) > 0
        same_doc_name = random.choice(same_docs)

        # select the documents that have the different value on specific key
        diff_doc_names = set()
        while len(diff_doc_names) < self.cand_doc_num - 1:
            tmp_doc_name = random.choice(self.all_docs_names)
            tmp_doc_kb = self.doc_reader.get_kb(tmp_doc_name, is_full=False)

            if self.slot in tmp_doc_kb and \
                    len(set(tmp_doc_kb[self.slot]).intersection(set(tar_doc_kb[self.slot]))) == 0:
                diff_doc_names.add(tmp_doc_name)

        cand_doc_names = list(diff_doc_names)
        cand_doc_names.append(same_doc_name)
        random.shuffle(cand_doc_names)  # shuffle

        # build candidate documents tensor
        ground_truth_idx = cand_doc_names.index(same_doc_name)
        cand_doc_tensor = self.doc_to_tensor(cand_doc_names,
                                             get_doc_id=self.doc_reader.get_doc_id,
                                             hierarchical=self.hierarchical)

        # build target document tensor(not for the similar document)
        tar_doc_tensor = self.doc_to_tensor([tar_doc_name],
                                            get_doc_id=self.doc_reader.get_doc_id,
                                            hierarchical=self.hierarchical)[0]
        return tar_doc_tensor, cand_doc_tensor, ground_truth_idx, cand_doc_names

    @staticmethod
    def collect_fun(batch):
        batch_tar_doc = []
        batch_cand_doc = []
        batch_ground_truth_idx = []
        # batch_cand_doc_names = []

        for ele in batch:
            batch_tar_doc.append(ele[0])
            batch_cand_doc.append(ele[1])
            batch_ground_truth_idx.append(ele[2])
            # batch_cand_doc_names.append(ele[3])

        batch_tar_doc = torch.stack(batch_tar_doc, dim=0)
        batch_cand_doc = torch.stack(batch_cand_doc, dim=0)
        batch_ground_truth_idx = torch.tensor(batch_ground_truth_idx, dtype=torch.long)

        return batch_tar_doc, batch_cand_doc, batch_ground_truth_idx
