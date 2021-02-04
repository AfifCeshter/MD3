#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = "Han"
__email__ = "liuhan132@foxmail.com"

import logging
from game.agent import Agent
from game.agent.state_module import HandCraftedState
from game.agent.policy_module import HandCraftedPolicy, RandPolicy, FixedPolicy, NonePolicy

logger = logging.getLogger(__name__)


class AgentRulesRules(Agent):
    """
    Agent with hand-crafted state and policy
    """

    def get_state_module(self):
        return HandCraftedState(self.game_config)

    def get_policy_module(self):
        return HandCraftedPolicy(self.game_config)


class AgentRulesNone(Agent):
    """
    Agent with hand-crafted state and policy
    """

    def get_state_module(self):
        return HandCraftedState(self.game_config)

    def get_policy_module(self):
        return NonePolicy(self.game_config)


class AgentRulesRand(Agent):
    """
    Agent with hand-crafted state and policy
    """

    def get_state_module(self):
        return HandCraftedState(self.game_config)

    def get_policy_module(self):
        return RandPolicy(self.game_config)


class AgentRulesFixed(Agent):
    """
    Agent with hand-crafted state and policy
    """

    def get_state_module(self):
        return HandCraftedState(self.game_config)

    def get_policy_module(self):
        return FixedPolicy(self.game_config)
