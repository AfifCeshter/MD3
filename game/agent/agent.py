#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = "Han"
__email__ = "liuhan132@foxmail.com"

import torch
import logging
from itertools import chain
from torch.distributions import Categorical
from game.agent.state_module import StateModule
from game.agent.policy_module import PolicyModule
from game.template import NLTemplate, AgentActs
from utils.functions import get_optimizer, count_parameters, get_entropy

logger = logging.getLogger(__name__)


class Agent:
    """
    Base Agent Module
    """

    def __init__(self, game_config, device):
        self.training = False
        self.game_config = game_config
        self.is_e2e = game_config['rl-train']['e2e']

        self.doc_threshold = game_config['global']['guess_thre']
        self.force_guess = game_config['global']['force_guess']
        self.max_turns = game_config['global']['max_turns']

        # initial config
        self.turn_i = 1
        self.act_his = []
        self.cand_docs = None
        self.cand_names = []
        self.dialog_level_doc_dist = None
        self.dialog_level_act_dist = None

        # device
        self.enable_cuda = torch.cuda.is_available()
        self.device = device

        # state module
        self.state_module = self.get_state_module().to(self.device)

        # policy module
        self.policy_module = self.get_policy_module().to(self.device)

        self.load_parameters()

        logger.info('Filter-Module Params=%d' % count_parameters(self.state_module))
        logger.info('Diff-Module Params=%d' % count_parameters(self.policy_module))

        # optimizer
        self.optimizer = None

        # NLG with template
        template_path = game_config['dataset']['agent_template_path']
        self.nl_template = NLTemplate(template_path)

    def get_state_module(self):
        """
        Need to implement in child class
        """
        return StateModule(self.game_config)

    def get_policy_module(self):
        """
        Need to implement in child class
        """
        return PolicyModule(self.game_config)

    def update(self, *args):
        """
        Need to implement in child class
        """
        return NotImplementedError

    def init_dialog(self, cand_docs, cand_docs_diff, cand_names):
        """
        Set arguments when dialog starting
        :param cand_docs:
        :param cand_docs_diff:
        :param cand_names:
        :return:
        """
        self.turn_i = 1
        self.act_his = []

        self.cand_docs = cand_docs
        self.cand_docs_diff = cand_docs_diff
        if isinstance(self.cand_docs, torch.Tensor):
            # (1, num_docs, num_slots, hidden_size * 4)
            self.cand_docs = self.cand_docs.unsqueeze(0).to(self.device)
        if isinstance(self.cand_docs_diff, torch.Tensor):
            self.cand_docs_diff = self.cand_docs_diff.unsqueeze(0).to(self.device)

        self.cand_names = cand_names  # (num_docs)

        # internal state
        self.dialog_level_doc_dist = torch.softmax(
            torch.ones(1, len(self.cand_names), dtype=torch.float, device=self.device),
            dim=-1)
        self.dialog_level_act_dist = torch.zeros(1, AgentActs.slot_size(), dtype=torch.float, device=self.device)

    def turn_act(self, last_turn):
        """
        Input with user answer in natural language
        and history in natural language
        :param dialog_state:
        :return:
        """
        if last_turn is not None:
            if isinstance(last_turn, torch.Tensor):
                last_turn = last_turn.to(self.device)

            # 1+2. NLU + DST module: dialog-level documents/actions distribution
            if not self.is_e2e:
                with torch.no_grad():
                    turn_level_doc_dist, turn_level_act_dist = self.state_module. \
                        get_turn_level_state(self.cand_docs, last_turn)
                    turn_level_doc_dist = turn_level_doc_dist.to(self.device)
                    turn_level_act_dist = turn_level_act_dist.to(self.device)
                    self.dialog_level_doc_dist, self.dialog_level_act_dist = self.state_module. \
                        update_dialog_level_state(self.dialog_level_doc_dist, turn_level_doc_dist,
                                                  self.dialog_level_act_dist, turn_level_act_dist)
            else:
                turn_level_doc_dist, turn_level_act_dist = self.state_module. \
                    get_turn_level_state(self.cand_docs, last_turn)
                self.dialog_level_doc_dist, self.dialog_level_act_dist = self.state_module. \
                    update_dialog_level_state(self.dialog_level_doc_dist, turn_level_doc_dist,
                                              self.dialog_level_act_dist, turn_level_act_dist)

        # 3. Policy module: generate action by documents difference
        agent_act_prob = self.policy_module.get_agent_act_prob(self.cand_docs_diff,
                                                               self.dialog_level_doc_dist,
                                                               self.dialog_level_act_dist,
                                                               self.turn_i)
        assert agent_act_prob.shape[0] == 1
        agent_act_prob = agent_act_prob.squeeze(0)

        # 4. guess or ask
        max_docs_prob, guess_doc_idx = self.dialog_level_doc_dist.max(dim=-1)
        max_docs_prob = max_docs_prob.cpu().item()
        # max_docs_prob = -get_entropy(self.dialog_level_doc_dist)
        agent_act = self.select_action(agent_act_prob, max_docs_prob)

        # guess the movie
        guess_doc_idx = guess_doc_idx.cpu().item()
        agent_value = self.cand_names[guess_doc_idx]

        # whether the first time to this action, used for natural language generating
        is_first = agent_act not in self.act_his
        agent_nl = self.nl_template.act_to_nl(agent_act, agent_value, is_first=is_first)

        self.act_his.append(agent_act)
        self.turn_i += 1

        return agent_act, agent_act_prob, agent_value, agent_nl

    def select_action(self, act_prob, max_docs_prob):
        """
        Select the agent action with greedy policy or probability distribution
        :param act_prop:
        :return:
        """
        _, action_id = act_prob.max(dim=-1)

        if max_docs_prob > self.doc_threshold or (self.force_guess and self.turn_i == self.max_turns):
            agent_act = AgentActs.GUESS
        else:
            if self.training:
                m = Categorical(act_prob)
                action_id = m.sample()
                agent_act_id = action_id.cpu().item()
                agent_act = AgentActs.id_to_slot(agent_act_id)
            else:
                agent_act_id = action_id.cpu().item()
                agent_act = AgentActs.id_to_slot(agent_act_id)
        return agent_act

    def save_parameters(self, num):
        """
        Save the trained parameters
        :param num:
        :return:
        """
        # if self.is_e2e:
        self.state_module.save_parameters(num)
        self.policy_module.save_parameters(num)

    def load_parameters(self):
        """
        Load the saved parameters
        :return:
        """
        self.state_module.load_parameters(self.enable_cuda)
        self.policy_module.load_parameters(self.enable_cuda)

    def get_parameters(self):
        if self.is_e2e:
            return chain(self.state_module.parameters(), self.policy_module.parameters())
        return self.policy_module.parameters()

    def train_config(self):
        self.training = True
        if self.is_e2e:
            self.state_module.train()
        else:
            self.state_module.eval()
        self.policy_module.train()

        # optimizer
        self.optimizer = get_optimizer(self.game_config['train']['optimizer'],
                                       self.game_config['train']['learning_rate'],
                                       self.get_parameters())

    def eval_config(self):
        self.training = False
        self.state_module.eval()
        self.policy_module.eval()
