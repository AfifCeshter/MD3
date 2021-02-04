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
from .doc_rep_pt_dataset import load_docs_rep
from game.template import NLTemplate, AgentActs
from game.agent.state_module import HandCraftedState
from game.simulator import SimulatorAct

logger = logging.getLogger(__name__)


class ImitationStateReader(BaseReader):
    """
    Dialog dataset reader for StateNet
    """

    def __init__(self, game_config):
        super(ImitationStateReader, self).__init__(game_config)

        self.use_all_docs = game_config['global']['all_docs']
        self.cand_doc_num = game_config['global']['cand_doc_num']

        # Hand-crafted state
        self.hand_crafted_state = HandCraftedState(game_config)

        # Agent NLG with template
        agent_template_path = game_config['dataset']['agent_template_path']
        self.agent_nl_template = NLTemplate(agent_template_path)

        # Simulator NLG with template
        simulator_template_path = game_config['dataset']['simulator_template_path']
        self.simulator_nl_template = NLTemplate(simulator_template_path)

    def _get_dataloader(self, batch_size, num_workers, iters, data_path, docs_rep_path_prefix):
        """
        Dialog data loader for supervised training
        :param docs_rep_path_prefix:
        :param data_path:
        :param batch_size:
        :return:
        """
        doc_dataset = ImitationStateDataset(self.turn_to_tensor,
                                            data_path,
                                            docs_rep_path_prefix,
                                            self.hand_crafted_state,
                                            self.agent_nl_template,
                                            self.simulator_nl_template,
                                            self.cand_doc_num,
                                            self.use_all_docs)
        cur_sampler = torch.utils.data.sampler.RandomSampler(doc_dataset,
                                                             replacement=True,
                                                             num_samples=iters * batch_size)
        return torch.utils.data.DataLoader(doc_dataset,
                                           batch_size=batch_size,
                                           sampler=cur_sampler,
                                           num_workers=num_workers,
                                           collate_fn=ImitationStateDataset.collect_fun)
        # return GenBatchSampleIter(doc_dataset,
        #                           batch_size,
        #                           DialogDocFilterDataset.collect_fun)

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


class ImitationStateDataset(torch.utils.data.Dataset):
    """
    Dialog dataset for StateNet
    """

    def __init__(self, turn_to_tensor, data_path, docs_rep_path_prefix, hand_crafted_state,
                 agent_nl_template, simulator_nl_template, cand_doc_num, use_all_docs):
        super(ImitationStateDataset, self).__init__()

        self.turn_to_tensor = turn_to_tensor

        # build documents dataset reader
        self.doc_reader = DocDataset(data_path)
        self.all_docs_names = self.doc_reader.get_all_names()

        # cand docs
        self.cand_doc_num = cand_doc_num if not use_all_docs else len(self.all_docs_names)

        self.docs_rep = load_docs_rep(docs_rep_path_prefix)

        # hand-crafted state
        self.hand_crafted_state = hand_crafted_state

        # nl template
        self.agent_template = agent_nl_template
        self.simulator_template = simulator_nl_template

    def __len__(self):
        return len(self.all_docs_names)

    def __getitem__(self, index):
        # 1. random sample candidate documents
        cand_docs_names = random.sample(self.all_docs_names, self.cand_doc_num)
        cand_docs_idx = [self.all_docs_names.index(name) for name in cand_docs_names]
        cand_docs_rep = self.docs_rep[cand_docs_idx, :]
        cand_kb = [self.doc_reader.get_kb(name, is_full=False) for name in cand_docs_names]

        # 2. random select target document
        tar_doc = random.choice(cand_docs_names)
        tar_doc_kb_full = self.doc_reader.get_kb(tar_doc, is_full=True)
        SimulatorAct.random_mask_(tar_doc_kb_full, rand_p=0.5)

        # 3.1 random generate agent action and user response
        valid_acts = list(tar_doc_kb_full.keys())
        none_acts = list(set(AgentActs.ALL_SLOTS).difference(set(valid_acts)))
        if len(none_acts) == 0 or (len(valid_acts) > 0 and random.random() > 0.5):
            agent_act = random.choice(valid_acts)
            user_value = random.choice(tar_doc_kb_full[agent_act])
        else:
            agent_act = random.choice(none_acts)
            user_value = None

        # # not balance no answer
        # agent_act = random.choice(AgentActs.ALL_SLOTS)
        # user_value = None
        # if agent_act in tar_doc_kb_full:
        #     user_value = random.choice(tar_doc_kb_full[agent_act])

        # 3.2 generate natural response
        is_first = True if random.random() > 0.5 else False
        agent_nl = self.agent_template.act_to_nl(agent_act, act_value=None, is_first=is_first)
        user_nl = self.simulator_template.act_to_nl(agent_act, user_value)

        # 3.3 build turn nl
        turn_tensor = self.turn_to_tensor(agent_nl, user_nl)

        # hand-crafted turn-level documents distribution
        turn_level_doc_dist, _, _ = self.hand_crafted_state.get_internal_state(cand_kb,
                                                                               (agent_act, user_value))

        assert turn_level_doc_dist.sum().item() > 0
        turn_level_doc_dist_gt = turn_level_doc_dist.gt(0).long()

        # user response classification
        turn_slot_gt = AgentActs.slot_to_id(agent_act)
        turn_inform_gt = 0. if user_value is not None else 1.

        return cand_docs_rep, turn_tensor, turn_level_doc_dist_gt, turn_slot_gt, turn_inform_gt

    @staticmethod
    def collect_fun(batch):
        batch_docs_rep = []
        batch_turn_tensor = []
        batch_valid_docs = []
        batch_user_slot_gt = []
        batch_user_inform_gt = []

        for ele in batch:
            batch_docs_rep.append(ele[0])
            batch_turn_tensor.append(ele[1])
            batch_valid_docs.append(ele[2])
            batch_user_slot_gt.append(ele[3])
            batch_user_inform_gt.append(ele[4])

        batch_docs_rep = torch.stack(batch_docs_rep, dim=0)
        batch_turn_tensor = torch.stack(batch_turn_tensor, dim=0)
        batch_valid_docs = torch.stack(batch_valid_docs, dim=0)
        batch_user_slot_gt = torch.tensor(batch_user_slot_gt, dtype=torch.long)
        batch_user_inform_gt = torch.tensor(batch_user_inform_gt, dtype=torch.float)

        return batch_docs_rep, batch_turn_tensor, batch_valid_docs, batch_user_slot_gt, batch_user_inform_gt
