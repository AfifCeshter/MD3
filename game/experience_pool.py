#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = "Han"
__email__ = "liuhan132@foxmail.com"

import torch
import random
from collections import namedtuple, deque
from utils.functions import to_long_tensor, to_float_tensor

# Define one step transition in dialog
Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'term'))


class ExperiencePool:
    """
    Experience Replay
    """

    def __init__(self, experience_pool_size):
        self.size = experience_pool_size
        self.buffer = deque(maxlen=experience_pool_size)

    def __len__(self):
        return self.size

    def add(self, state, action, reward, next_state, terminal):
        example = Transition(state, action, reward, next_state, terminal)
        self.buffer.append(example)

    def _encode_sample(self, idx):
        batch_states = []
        batch_actions = []
        batch_rewards = []
        batch_next_states = []
        batch_terminals = []

        for i in idx:
            data = self.buffer[i]
            batch_states.append(data.state)
            batch_actions.append(data.action)
            batch_rewards.append(data.reward)
            batch_next_states.append(data.next_state)
            batch_terminals.append(data.term)

        batch_states_tensor = self._statck_state(batch_states)
        batch_actions_tensor = to_long_tensor(batch_actions)
        batch_rewards_tensor = to_float_tensor(batch_rewards)
        batch_next_states_tensor = self._statck_state(batch_next_states)
        batch_terminals_tensor = to_long_tensor(batch_terminals)

        return Transition(batch_states_tensor, batch_actions_tensor, batch_rewards_tensor,
                          batch_next_states_tensor, batch_terminals_tensor)

    def sample(self, batch_size):
        idxes = [random.randint(0, len(self.buffer) - 1) for _ in range(batch_size)]
        return self._encode_sample(idxes)

    def _statck_state(self, list_states):
        """
        stack the list of states to batch array
        :param list_states:
        :return:
        """
        batch_last_docs_prob = []
        batch_last_turn = []

        for ele in list_states:
            batch_last_docs_prob.append(ele.last_docs_prob)
            batch_last_turn.append(ele.last_turn)

        # (batch, num_docs)
        batch_last_docs_prob = torch.cat(batch_last_docs_prob, dim=0)
        # (batch, turn_len)
        batch_last_turn = torch.cat(batch_last_turn, dim=0)

        return DialogState(batch_last_docs_prob, batch_last_turn)
