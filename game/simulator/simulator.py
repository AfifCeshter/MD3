#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = "Han"
__email__ = "liuhan132@foxmail.com"


class Simulator:
    def __init__(self):
        pass

    def init_dialog(self, *args):
        return NotImplementedError

    def respond_act(self, agent_act, agent_value):
        return NotImplementedError

    def respond_nl(self, agent_nl):
        return NotImplementedError
