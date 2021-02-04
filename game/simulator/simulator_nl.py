#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = "Han"
__email__ = "liuhan132@foxmail.com"


from game.simulator import Simulator


class SimulatorNL(Simulator):
    def __init__(self):
        super(SimulatorNL, self).__init__()

    def init_dialog(self, *args):
        pass

    def respond_act(self, agent_act, agent_value):
        pass

    def respond_nl(self, agent_nl):
        pass