#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = "Han"
__email__ = "liuhan132@foxmail.com"

import random
import logging
import torch
import torch.utils.data
from .base_reader import BaseReader
from .doc_dataset import DocDataset
from .doc_rep_pt_dataset import load_docs_rep, get_docs_diff_rep
from game.agent.policy_module import HandCraftedPolicy
from game.template import AgentActs
from utils.functions import del_zeros_right

logger = logging.getLogger(__name__)


class ImitationPolicyReader(BaseReader):
    """
    Dataset for Imitation Learning on policy-net
    """

    def __init__(self, game_config):
        super(ImitationPolicyReader, self).__init__(game_config)
        self.cand_doc_num = game_config['global']['cand_doc_num']

        # Hand-crafted policy
        self.hand_crafted_policy = HandCraftedPolicy(game_config)

    def _get_dataloader(self, batch_size, num_workers, iters, data_path, docs_rep_path_prefix):
        """
        Dialog data loader for supervised training
        :param docs_rep_path_prefix:
        :param data_path:
        :param batch_size:
        :return:
        """
        doc_dataset = ImitationPolicyDataset(data_path,
                                             self.cand_doc_num,
                                             docs_rep_path_prefix,
                                             self.hand_crafted_policy)
        cur_sampler = torch.utils.data.sampler.RandomSampler(doc_dataset,
                                                             replacement=True,
                                                             num_samples=iters * batch_size)
        return torch.utils.data.DataLoader(doc_dataset,
                                           batch_size=batch_size,
                                           sampler=cur_sampler,
                                           num_workers=num_workers,
                                           collate_fn=ImitationPolicyDataset.collect_fun)

    def get_dataloader_train(self, batch_size, num_workers, iters):
        return self._get_dataloader(batch_size,
                                    num_workers=num_workers,
                                    iters=iters,
                                    data_path=self.data_prefix + 'doc_rep/guessmovie_rep_doc_id.json',
                                    docs_rep_path_prefix=self.data_prefix + 'doc_rep/pt/rep_doc_pt_rep.pt')

    def get_dataloader_test(self, batch_size, num_workers, iters):
        return self._get_dataloader(batch_size,
                                    num_workers=num_workers,
                                    iters=iters,
                                    data_path=self.data_prefix + 'guessmovie_dialog_doc_id.json',
                                    docs_rep_path_prefix=self.data_prefix + 'doc_rep/pt/dialog_doc_pt_rep.pt')


class ImitationPolicyDataset(torch.utils.data.Dataset):
    """
    Dataset for Imitation Learning on policy-net
    """

    def __init__(self, data_path, cand_doc_num, docs_rep_path_prefix, hand_crafted_policy):
        super(ImitationPolicyDataset, self).__init__()

        self.cand_doc_num = cand_doc_num
        self.hand_crafted_policy = hand_crafted_policy

        # build documents dataset reader
        self.doc_reader = DocDataset(data_path)
        self.all_docs_names = self.doc_reader.get_all_names()

        # documents representation
        self.docs_rep = load_docs_rep(docs_rep_path_prefix)
        self.docs_diff_rep = get_docs_diff_rep(self.docs_rep)

    def __len__(self):
        return len(self.all_docs_names)

    def __getitem__(self, index):
        # 1. Candidate documents
        cand_docs_idx = random.sample(range(len(self.all_docs_names)), self.cand_doc_num)
        cand_docs_names = [self.all_docs_names[i] for i in cand_docs_idx]
        cand_kb = [self.doc_reader.get_kb(name, is_full=False) for name in cand_docs_names]
        cand_docs_diff_rep = self.docs_diff_rep[cand_docs_idx, :]

        # 2. Dialog-Level state
        dialog_level_doc_dist = torch.ones(self.cand_doc_num, dtype=torch.float)
        doc_num = int(random.random() * self.cand_doc_num)
        if doc_num:
            doc_idx = random.sample(range(self.cand_doc_num), doc_num)
            dialog_level_doc_dist[doc_idx] = 0
        dialog_level_doc_dist = torch.softmax(dialog_level_doc_dist, dim=-1).unsqueeze(0)

        dialog_level_act_dist = torch.zeros(AgentActs.slot_size(), dtype=torch.float)
        act_num = int(random.random() * AgentActs.slot_size())
        if act_num:
            act_idx = random.sample(range(AgentActs.slot_size()), act_num)
            dialog_level_act_dist[act_idx] = 1
        dialog_level_act_dist = torch.softmax(dialog_level_act_dist, dim=-1).unsqueeze(0)

        # 2. Get the HandCrafted policy action
        agent_act_prob = self.hand_crafted_policy.get_agent_act_prob(cand_kb,
                                                                     dialog_level_doc_dist,
                                                                     dialog_level_act_dist, None)

        # 3. Get the ground truth agent act and value
        _, agent_act_id = agent_act_prob.max(dim=-1)

        return cand_docs_diff_rep, dialog_level_doc_dist, dialog_level_act_dist, agent_act_id

    @staticmethod
    def collect_fun(batch):
        batch_docs_diff_rep = []
        dialog_level_doc_dist = []
        dialog_level_act_dist = []
        batch_agent_act_gt = []

        for ele in batch:
            batch_docs_diff_rep.append(ele[0])
            dialog_level_doc_dist.append(ele[1])
            dialog_level_act_dist.append(ele[2])
            batch_agent_act_gt.append(ele[3])

        batch_docs_diff_rep = torch.stack(batch_docs_diff_rep, dim=0)
        dialog_level_doc_dist = torch.cat(dialog_level_doc_dist, dim=0)
        dialog_level_act_dist = torch.cat(dialog_level_act_dist, dim=0)
        batch_agent_act_gt = torch.tensor(batch_agent_act_gt, dtype=torch.long)

        return batch_docs_diff_rep, dialog_level_doc_dist, dialog_level_act_dist, batch_agent_act_gt
