#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = "Han"
__email__ = "liuhan132@foxmail.com"

import numpy as np
import torch
from game.agent import Agent
from game.agent.state_module import MRCState
from game.agent.policy_module import RandPolicy, FixedPolicy, NeuralPolicy


class AgentMRCRand(Agent):
    """
    Agent MRC Module
    """

    def get_state_module(self):
        return MRCState(self.game_config)

    def get_policy_module(self):
        return RandPolicy(self.game_config)


class AgentMRCFixed(Agent):
    """
    Agent MRC Module
    """

    def get_state_module(self):
        return MRCState(self.game_config)

    def get_policy_module(self):
        return FixedPolicy(self.game_config)


class AgentMRCModel(Agent):
    """
    Agent MRC Module
    """

    def __init__(self, game_config, device):
        super(AgentMRCModel, self).__init__(game_config, device)
        self.eps = np.finfo(np.float32).eps.item()

        self.gamma = game_config['rl-train']['gamma']
        self.epsilon = game_config['rl-train']['epsilon']

    def get_state_module(self):
        return MRCState(self.game_config)

    def get_policy_module(self):
        return NeuralPolicy(self.game_config)

    def update(self, rewards, saved_log_probs, saved_log_doc_probs=None):
        if len(saved_log_probs) == 0:
            return 0.

        assert len(rewards) == len(saved_log_probs)

        R = 0
        policy_loss = []
        returns = []
        for r in rewards[::-1]:
            R = r + self.gamma * R
            returns.insert(0, R)

        for log_prob, R in zip(saved_log_probs, returns):
            policy_loss.append(-log_prob * R)

        self.optimizer.zero_grad()
        policy_loss = torch.stack(policy_loss).sum()
        policy_loss.backward()
        self.optimizer.step()

        return policy_loss.cpu().item()
