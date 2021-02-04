#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = "Han"
__email__ = "liuhan132@foxmail.com"

import os
import random
import logging
import torch
from game.models import PTPolicyNet
from game.template import AgentActs
from utils.functions import save_model, load_checkpoint_parameters

logger = logging.getLogger(__name__)


class PolicyModule(torch.nn.Module):
    def __init__(self, game_config):
        super(PolicyModule, self).__init__()
        self.game_config = game_config

        self.force_guess = game_config['global']['force_guess']
        self.max_turns = game_config['global']['max_turns']

    def load_parameters(self, enable_cuda, force=False, strict=False):
        pass

    def save_parameters(self, num):
        pass

    def get_agent_act_prob(self, cand_docs, dialog_level_doc_dist, dialog_level_act_dist, turn_i):
        """
        Policy Module
        """
        return NotImplementedError


class NonePolicy(PolicyModule):
    def get_agent_act_prob(self, cand_docs, dialog_level_doc_dist, dialog_level_act_dist, turn_i):
        return torch.softmax(1 - dialog_level_act_dist, dim=-1)


class HandCraftedPolicy(PolicyModule):

    def __init__(self, game_config):
        super(HandCraftedPolicy, self).__init__(game_config)

        self.enable_full_answer = game_config['global']['full_answer']
        self.no_inform_threshold = 0.3
        self.doc_threshold = 0

    def get_agent_act_prob(self, cand_kb, dialog_level_doc_dist, dialog_level_act_dist, turn_i):
        # filter slots that have no answer
        is_no_inform_slot = dialog_level_act_dist.gt(self.no_inform_threshold)[0].cpu().tolist()
        all_no_ans_slots = [AgentActs.id_to_slot(i) for i, flag in enumerate(is_no_inform_slot) if flag]
        # all_no_ans_slots = []     # for no-dst used

        # filter the documents that was excluded
        valid_docs = dialog_level_doc_dist.gt(self.doc_threshold)[0].cpu().tolist()
        valid_docs_kb = [cand_kb[i] for i, flag in enumerate(valid_docs) if flag]
        cand_kb_slots = self.convert_kb(valid_docs_kb)

        # select the slots with minimal different values along all candidate KB
        agent_act = AgentActs.DIRECTED_BY
        max_num_values = 0
        min_num_values = 1e+4

        for slot_name, slot_value in cand_kb_slots.items():

            # already asked, no answer
            if slot_name in all_no_ans_slots:
                continue

            # get different values without none
            set_slot_value = set()
            for v in slot_value:
                if v is None:
                    continue

                set_slot_value.add(str(v))

            # has more than 2 different values
            # if 1 < len(set_slot_value) < min_num_values:
            #     agent_act = slot_name
            #     min_num_values = len(set_slot_value)

            if len(set_slot_value) > max_num_values:
                agent_act = slot_name
                max_num_values = len(set_slot_value)

        # convert to distributions
        agent_act_id = AgentActs.slot_to_id(agent_act)
        agent_act_prob = dialog_level_act_dist.new_zeros(1, AgentActs.slot_size())
        agent_act_prob[0][agent_act_id] = 1.0

        return agent_act_prob

    def convert_kb(self, cand_kb):
        """
        convert the KB with subjects to KB with relations
        :param cand_kb: KB
        :return:
        """
        kb_slots = {}
        for kb in cand_kb:
            for slot in AgentActs.ALL_SLOTS:
                if slot not in kb_slots:
                    kb_slots[slot] = []

                if slot in kb:
                    kb_slots[slot].append(kb[slot])
                else:
                    kb_slots[slot].append(None)

        return kb_slots


class RandPolicy(PolicyModule):
    def get_agent_act_prob(self, cand_docs, dialog_level_doc_dist, dialog_level_act_dist, turn_i):
        agent_act_id = random.randrange(AgentActs.slot_size())
        agent_act_prob = dialog_level_doc_dist.new_zeros(1, AgentActs.slot_size())
        agent_act_prob[0][agent_act_id] = 1.0

        return agent_act_prob


class FixedPolicy(PolicyModule):
    def __init__(self, game_config):
        super(FixedPolicy, self).__init__(game_config=game_config)
        self.fixed_policy = ['release_year', 'directed_by', 'in_language',
                             'written_by', 'starred_actors', 'has_genre']

    def get_agent_act_prob(self, cand_docs, dialog_level_doc_dist, dialog_level_act_dist, turn_i):
        agent_act_id = turn_i   # default fixed policy
        # agent_act_id = AgentActs.slot_to_id(self.fixed_policy[turn_i-1])

        agent_act_prob = dialog_level_doc_dist.new_zeros(1, AgentActs.slot_size())
        agent_act_prob[0][agent_act_id % AgentActs.slot_size()] = 1.0

        return agent_act_prob


class NeuralPolicy(PolicyModule):

    def __init__(self, game_config):
        super(NeuralPolicy, self).__init__(game_config)
        self.in_policy_weight_path = game_config['checkpoint']['in_policy_weight_path']
        self.in_policy_checkpoint_path = game_config['checkpoint']['in_policy_checkpoint_path']
        self.out_policy_weight_path = game_config['checkpoint']['out_policy_weight_path']
        self.out_policy_checkpoint_path = game_config['checkpoint']['out_policy_checkpoint_path']

        self.policy_model = PTPolicyNet(game_config['model'])

    def load_parameters(self, enable_cuda, force=False, strict=False):
        if force:
            assert os.path.exists(self.in_policy_checkpoint_path)

        if os.path.exists(self.in_policy_checkpoint_path):
            logger.info('Loading agent parameters for policy module')
            load_weight_path = load_checkpoint_parameters(self,
                                                          self.in_policy_weight_path,
                                                          self.in_policy_checkpoint_path,
                                                          enable_cuda,
                                                          strict)
            logger.info('Loaded policy module from %s' % load_weight_path)

    def save_parameters(self, num):
        """
        Save the trained parameters
        :param num:
        :return:
        """
        logger.info('Saving agent parameters for policy module on step=%d' % num)
        save_model(self,
                   num,
                   model_weight_path=self.out_policy_weight_path + '-' + str(num),
                   checkpoint_path=self.out_policy_checkpoint_path)

    def get_agent_act_prob(self, cand_docs, dialog_level_doc_dist, dialog_level_act_dist, turn_i):
        agent_act_prob = self.policy_model(cand_docs, dialog_level_doc_dist, dialog_level_act_dist)

        return agent_act_prob
