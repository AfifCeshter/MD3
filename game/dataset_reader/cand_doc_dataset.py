#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = "Han"
__email__ = "liuhan132@foxmail.com"

import json
import random
import logging
import torch.utils.data
from game.template import AgentActs
from .base_reader import BaseReader
from .doc_rep_pt_dataset import load_docs_rep, get_docs_diff_rep

logger = logging.getLogger(__name__)


class BaseCandDocReader(BaseReader):
    """
    Base candidates documents reader
    """

    def __init__(self, game_config):
        super(BaseCandDocReader, self).__init__(game_config)
        self.use_all_docs = game_config['global']['all_docs']
        self.cand_doc_num = game_config['global']['cand_doc_num'] if not self.use_all_docs else None
        self.cand_doc_path = self.dataset_config['cand_doc_path'] if not self.use_all_docs else None

    def get_dataset(self):
        return NotImplementedError

    def get_fixed_dataset(self):
        return NotImplementedError

    def get_dataset_reader(self, num_workers, iters):
        doc_dataset = self.get_dataset()
        cur_sampler = torch.utils.data.sampler.RandomSampler(doc_dataset,
                                                             replacement=True,
                                                             num_samples=iters)

        return torch.utils.data.DataLoader(doc_dataset,
                                           batch_size=1,
                                           sampler=cur_sampler,
                                           num_workers=num_workers,
                                           collate_fn=RandDocDataset.collect_fun)

    def get_fixed_reader(self, num_workers):
        doc_dataset = self.get_fixed_dataset()
        return torch.utils.data.DataLoader(doc_dataset,
                                           batch_size=1,
                                           shuffle=False,
                                           num_workers=num_workers,
                                           collate_fn=RandDocDataset.collect_fun)


class CandDocDocReader(BaseCandDocReader):
    """
    Candidates documents + documents reader
    """

    def __init__(self, game_config):
        super(CandDocDocReader, self).__init__(game_config)
        self.docs_rep = load_docs_rep(self.data_prefix + 'doc_rep/pt/dialog_doc_pt_rep.pt')
        self.docs_diff_rep = get_docs_diff_rep(self.docs_rep)

    def get_dataset(self):
        return RandDocDataset(self.cand_doc_num,
                              self.doc_reader.get_all_names(),
                              self.docs_rep,
                              self.docs_diff_rep,
                              self.doc_reader.get_kb)

    def get_fixed_dataset(self):
        return FixedDocDataset(self.cand_doc_path,
                               self.doc_reader.get_all_names(),
                               self.docs_rep,
                               self.docs_diff_rep,
                               self.doc_reader.get_kb)


class CandDocKBReader(BaseCandDocReader):
    """
    Candidates documents + KB reader
    """

    def __init__(self, game_config):
        super(CandDocKBReader, self).__init__(game_config)
        self.docs_rep = load_docs_rep(self.data_prefix + 'doc_rep/pt/dialog_doc_pt_rep.pt')
        self.docs_diff_rep = get_docs_diff_rep(self.docs_rep)

    def get_dataset(self):
        return RandDocDataset(self.cand_doc_num,
                              self.doc_reader.get_all_names(),
                              self.docs_rep,
                              self.doc_reader.get_all_kb(),
                              self.doc_reader.get_kb)

    def get_fixed_dataset(self):
        return FixedDocDataset(self.cand_doc_path,
                               self.doc_reader.get_all_names(),
                               self.docs_rep,
                               self.doc_reader.get_all_kb(),
                               self.doc_reader.get_kb)


class CandKBKBReader(BaseCandDocReader):
    """
    Candidates KB + KB reader
    """

    def get_dataset(self):
        return RandDocDataset(self.cand_doc_num,
                              self.doc_reader.get_all_names(),
                              self.doc_reader.get_all_kb(),
                              self.doc_reader.get_all_kb(),
                              self.doc_reader.get_kb)

    def get_fixed_dataset(self):
        return FixedDocDataset(self.cand_doc_path,
                               self.doc_reader.get_all_names(),
                               self.doc_reader.get_all_kb(),
                               self.doc_reader.get_all_kb(),
                               self.doc_reader.get_kb)


class CandMRCMRCReader(BaseCandDocReader):
    """
    Candidiate MRC + MRC reader
    """

    def __init__(self, game_config):
        super(CandMRCMRCReader, self).__init__(game_config)

        self._doc_mrc_path = game_config['dataset']['data_prefix'] + \
                             '/mrc_doc/mrc_no_answer/guessmovie_dialog_mrc_ans.json'
        self._doc_mrc_data = self.read_mrc_data()

    def read_mrc_data(self):
        with open(self._doc_mrc_path, 'r') as f:
            doc_mrc_data = json.load(f)

        doc_mrc_dict = {}
        for ele in doc_mrc_data:
            doc_mrc_dict[ele['name']] = ele['mrc_kb']

        return doc_mrc_dict

    def docs_to_mrckb(self, cand_names):
        return [self._doc_mrc_data[name.lower()] for name in cand_names]

    def get_dataset(self):
        return RandDocDataset(self.cand_doc_num,
                              list(self._doc_mrc_data.keys()),
                              list(self._doc_mrc_data.values()),
                              list(self._doc_mrc_data.values()),
                              self.doc_reader.get_kb)

    def get_fixed_dataset(self):
        return FixedDocDataset(self.cand_doc_path,
                               list(self._doc_mrc_data.keys()),
                               list(self._doc_mrc_data.values()),
                               list(self._doc_mrc_data.values()),
                               self.doc_reader.get_kb)


class CandMRCDocReader(CandMRCMRCReader):
    """
    Candidiate MRC + documents reader
    """
    def __init__(self, game_config):
        super(CandMRCDocReader, self).__init__(game_config)
        self.docs_rep = load_docs_rep(self.data_prefix + 'doc_rep/pt/dialog_doc_pt_rep.pt')
        self.docs_diff_rep = get_docs_diff_rep(self.docs_rep)

    def get_dataset(self):
        return RandDocDataset(self.cand_doc_num,
                              list(self._doc_mrc_data.keys()),
                              list(self._doc_mrc_data.values()),
                              self.docs_diff_rep,
                              self.doc_reader.get_kb)

    def get_fixed_dataset(self):
        return FixedDocDataset(self.cand_doc_path,
                               list(self._doc_mrc_data.keys()),
                               list(self._doc_mrc_data.values()),
                               self.docs_diff_rep,
                               self.doc_reader.get_kb)


class FixedDocDataset(torch.utils.data.Dataset):
    """
    Candidate documents dataset
    """

    def __init__(self, cand_doc_path, all_doc_names, all_doc_rep, all_doc_diff_rep, doc_to_kb):
        super(FixedDocDataset, self).__init__()
        self.cand_doc_path = cand_doc_path
        self.all_doc_names = all_doc_names
        self.all_doc_rep = all_doc_rep

        self.all_doc_diff_rep = all_doc_diff_rep
        self.doc_to_kb = doc_to_kb

        # read candidate documents name
        with open(self.cand_doc_path, 'r') as f:
            self.cand_doc_name = json.load(f)
        if len(self.cand_doc_name) > 500:
            self.cand_doc_name = self.cand_doc_name[:500]

    def __getitem__(self, index):
        cand_names_json = self.cand_doc_name[index]

        # candidate documents
        cand_names = cand_names_json['cand_names']
        cand_docs_idx = [self.all_doc_names.index(name) for name in cand_names]

        if isinstance(self.all_doc_rep, list):
            cand_docs_rep = [self.all_doc_rep[i] for i in cand_docs_idx]
        else:
            cand_docs_rep = self.all_doc_rep[cand_docs_idx, :]

        if isinstance(self.all_doc_diff_rep, list):
            cand_docs_diff_rep = [self.all_doc_diff_rep[i] for i in cand_docs_idx]
        else:
            cand_docs_diff_rep = self.all_doc_diff_rep[cand_docs_idx, :]

        # target documents
        tar_name = cand_names_json['tar_name']  # 'name' or 'tar_name'
        tar_kb = self.doc_to_kb(tar_name, is_full=True)

        return cand_docs_rep, cand_docs_diff_rep, cand_names, tar_kb, tar_name

    def __len__(self):
        return len(self.cand_doc_name)

    @staticmethod
    def collect_fun(batch):
        assert len(batch) == 1
        return batch[0]


class RandDocDataset(torch.utils.data.Dataset):
    """
    All documents as candidates dataset
    """

    def __init__(self, cand_doc_num, all_doc_names, all_doc_rep, all_doc_diff_rep, doc_to_kb):
        super(RandDocDataset, self).__init__()
        self.all_doc_names = all_doc_names
        self.all_doc_rep = all_doc_rep
        self.all_doc_diff_rep = all_doc_diff_rep
        self.doc_to_kb = doc_to_kb
        self.cand_doc_num = cand_doc_num

        # with open(doc_rep_same_name_path, 'r') as f:
        #     self.doc_dialog_same_name_data = json.load(f)
        #
        # with open(doc_rep_unk_name_path, 'r') as f:
        #     self.doc_dialog_unk_name_data = json.load(f)

    def __len__(self):
        return len(self.all_doc_rep)

    def __getitem__(self, item):
        # select from all documents
        if self.cand_doc_num is None:
            cand_names = self.all_doc_names
            cand_docs_rep = self.all_doc_rep
            cand_docs_diff_rep = self.all_doc_diff_rep

            tar_name = self.all_doc_names[item]

        # random sample from all docs
        else:
            # random mask a slot on candidate documents
            # same_slot = random.choice(AgentActs.ALL_SLOTS)
            # same_value_docs = random.choice(list(self.doc_dialog_same_name_data[same_slot].items()))
            # cand_names = same_value_docs[1]
            #
            # if len(cand_names) < self.cand_doc_num:
            #     cand_names += random.sample(self.doc_dialog_unk_name_data[same_slot],
            #                                 self.cand_doc_num - len(cand_names))
            # else:
            #     cand_names = random.sample(cand_names, self.cand_doc_num)

            cand_names = random.sample(self.all_doc_names, self.cand_doc_num)
            cand_docs_idx = [self.all_doc_names.index(name) for name in cand_names]

            if isinstance(self.all_doc_rep, list):
                cand_docs_rep = [self.all_doc_rep[i] for i in cand_docs_idx]
            else:
                cand_docs_rep = self.all_doc_rep[cand_docs_idx, :]

            if isinstance(self.all_doc_diff_rep, list):
                cand_docs_diff_rep = [self.all_doc_diff_rep[i] for i in cand_docs_idx]
            else:
                cand_docs_diff_rep = self.all_doc_diff_rep[cand_docs_idx, :]

            tar_name = random.choice(cand_names)

        tar_kb = self.doc_to_kb(tar_name, is_full=True)

        return cand_docs_rep, cand_docs_diff_rep, cand_names, tar_kb, tar_name

    @staticmethod
    def collect_fun(batch):
        assert len(batch) == 1
        return batch[0]
