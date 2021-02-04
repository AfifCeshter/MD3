#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = "Han"
__email__ = "liuhan132@foxmail.com"

import os
import logging
import nltk
from collections import Counter
import torch
import torch.nn.functional as F
from game.models import PTNLU
from game.template import AgentActs
from utils.functions import save_model, load_checkpoint_parameters

logger = logging.getLogger(__name__)


class StateModule(torch.nn.Module):
    def __init__(self, game_config):
        super(StateModule, self).__init__()
        self.game_config = game_config
        self.dialog_his_pro = game_config['global']['mask_his']

    def load_parameters(self, enable_cuda, force=False, strict=False):
        pass

    def save_parameters(self, num):
        pass

    def get_turn_level_state(self, cand_docs, last_turn):
        turn_level_doc_dist, turn_slot_cls, turn_inform_sig = self.get_internal_state(cand_docs, last_turn)
        turn_slot_inform = turn_slot_cls * turn_inform_sig + self.dialog_his_pro * turn_slot_cls
        turn_slot_inform = turn_slot_inform.clamp(min=0, max=1)

        return turn_level_doc_dist, turn_slot_inform

    def get_internal_state(self, cand_docs, last_turn):
        """
        Internal state
        :param cand_docs:
        :param last_turn:
        :return:
        """
        return NotImplementedError

    def update_dialog_level_state(self, last_dialog_level_doc_dist, turn_level_doc_dist,
                                  last_dialog_level_act_dist, turn_level_act_dist):
        dialog_level_doc_dist = DSTModule.update_dialog_level_doc_dist(last_dialog_level_doc_dist,
                                                                       turn_level_doc_dist)
        dialog_level_act_dist = DSTModule.update_dialog_level_act_dist(last_dialog_level_act_dist,
                                                                       turn_level_act_dist)
        return dialog_level_doc_dist, dialog_level_act_dist


class HandCraftedState(StateModule):
    """
    Hand-Crafted state module
    """

    def __init__(self, game_config):
        super(HandCraftedState, self).__init__(game_config)
        self.enable_full_answer = game_config['global']['full_answer']

    def get_internal_state(self, cand_kb, last_turn):
        agent_act, user_value = last_turn

        # slot cls and inform cls
        agent_act_id = AgentActs.slot_to_id(agent_act)
        turn_slot_cls = torch.zeros(AgentActs.slot_size())
        turn_slot_cls[agent_act_id] = 1

        # user have no answer
        if user_value is None:
            turn_inform_cls = 1
            turn_level_doc_dist = [1 for _ in range(len(cand_kb))]
        else:
            turn_inform_cls = 0
            turn_level_doc_dist = []
            for kb in cand_kb:
                if not self.enable_full_answer:
                    if agent_act not in kb or user_value in kb[agent_act]:
                        turn_level_doc_dist.append(1)
                        continue
                else:
                    if ', ' in user_value:
                        user_value = user_value.split(', ')

                    if agent_act not in kb or kb[agent_act] == user_value or kb[agent_act] == [user_value]:
                        turn_level_doc_dist.append(1)
                        continue

                turn_level_doc_dist.append(float('-inf'))

        turn_level_doc_dist = torch.softmax(torch.tensor(turn_level_doc_dist, dtype=torch.float), dim=-1)
        return turn_level_doc_dist, turn_slot_cls, turn_inform_cls


class MRCState(StateModule):
    """
    MRC State Module
    """
    def get_internal_state(self, cand_kb, last_turn):
        agent_act, user_value = last_turn

        # slot cls and inform cls
        agent_act_id = AgentActs.slot_to_id(agent_act)
        turn_slot_cls = torch.zeros(AgentActs.slot_size())
        turn_slot_cls[agent_act_id] = 1

        # user have no answer
        if user_value is None:
            turn_inform_cls = 1
            turn_level_doc_dist = [1 for _ in range(len(cand_kb))]
        else:
            turn_inform_cls = 0
            turn_level_doc_dist = []

            for kb in cand_kb:
                if kb[agent_act] == '':
                    score = 1
                    turn_inform_cls = 1
                else:
                    score = self.fake_f1_score(kb[agent_act], user_value)[0]
                turn_level_doc_dist.append(score)

        turn_level_doc_dist = torch.softmax(torch.tensor(turn_level_doc_dist, dtype=torch.float), dim=-1)
        return turn_level_doc_dist, turn_slot_cls, turn_inform_cls

    def fake_f1_score(self, a_sen, b_sen):
        a_tokens = nltk.word_tokenize(a_sen)
        b_tokens = nltk.word_tokenize(b_sen)

        if len(a_tokens) == 0 or len(b_tokens) == 0:
            return 1, 1, 1

        common = Counter(a_tokens) & Counter(b_tokens)
        num_same = sum(common.values())
        if num_same == 0:
            return 0, 0, 0
        precision = 1.0 * num_same / len(a_tokens)
        recall = 1.0 * num_same / len(b_tokens)
        f1 = (2 * precision * recall) / (precision + recall)
        return precision, recall, f1


class NeuralState(StateModule):
    """
    Neural state module
    """

    def __init__(self, game_config):
        super(NeuralState, self).__init__(game_config)
        self.in_state_weight_path = game_config['checkpoint']['in_state_weight_path']
        self.in_state_checkpoint_path = game_config['checkpoint']['in_state_checkpoint_path']
        self.out_state_weight_path = game_config['checkpoint']['out_state_weight_path']
        self.out_state_checkpoint_path = game_config['checkpoint']['out_state_checkpoint_path']

        self.state_nlu = PTNLU(game_config['model'],
                               embedding_path=game_config['dataset']['embedding_path'],
                               embedding_freeze=game_config['dataset']['embedding_freeze'])

    def get_internal_state(self, cand_docs, last_turn):
        return self.state_nlu(cand_docs, last_turn)

    def load_parameters(self, enable_cuda, force=False, strict=False):
        if force:
            assert os.path.exists(self.in_state_checkpoint_path)

        if os.path.exists(self.in_state_checkpoint_path):
            logger.info('Loading agent parameters for state module')
            load_weight_path = load_checkpoint_parameters(self,
                                                          self.in_state_weight_path,
                                                          self.in_state_checkpoint_path,
                                                          enable_cuda,
                                                          strict)
            logger.info('Loaded state module from %s' % load_weight_path)

    def save_parameters(self, num):
        """
        Save the trained parameters
        :param num:
        :return:
        """
        logger.info('Saving agent parameters for state module on step=%d' % num)
        save_model(self,
                   num,
                   model_weight_path=self.out_state_weight_path + '-' + str(num),
                   checkpoint_path=self.out_state_checkpoint_path)


class DSTModule:

    @staticmethod
    def update_dialog_level_doc_dist(last_dialog_level_doc_dist, turn_level_doc_dist):
        """
        DST-2 Module
            Update the dialog-level document distribution:
            1. multiply last dialog-level documents distribution and current turn-level documents distribution
            2. normalization with 2-norm

        Args:
            last_dialog_level_doc_dist: (batch, num_docs)
            turn_level_doc_dist: (batch, num_docs)
        """
        # new_dialog_level_doc_dist = 0.5 * last_dialog_level_doc_dist + 0.5 * turn_level_doc_dist

        new_dialog_level_doc_dist = last_dialog_level_doc_dist * turn_level_doc_dist
        # new_dialog_level_doc_dist = new_dialog_level_doc_dist.clamp(min=0, max=1)
        new_dialog_level_doc_dist = F.normalize(new_dialog_level_doc_dist, p=1, dim=-1)

        return new_dialog_level_doc_dist

    @staticmethod
    def update_dialog_level_act_dist(last_dialog_level_act_dist, turn_level_act_dist):
        new_dialog_level_act_dist = last_dialog_level_act_dist + turn_level_act_dist
        # new_dialog_level_act_dist = F.normalize(new_dialog_level_act_dist, p=1, dim=-1)
        new_dialog_level_act_dist = new_dialog_level_act_dist.clamp(min=0, max=1).round()

        return new_dialog_level_act_dist
