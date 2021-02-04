#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = "Han"
__email__ = "liuhan132@foxmail.com"

import copy
import random
from game.simulator import Simulator
from game.template import AgentActs, NLTemplate


class SimulatorAct(Simulator):
    def __init__(self, template_path, enable_full_answer, rand_p):
        super(SimulatorAct, self).__init__()
        self.target_name = None
        self.target_kb = None
        self.enable_full_answer = enable_full_answer
        self.rand_p = rand_p

        self.nl_template = NLTemplate(template_path)

    @staticmethod
    def random_mask_(target_kb, rand_p):
        # random mask some value in slots
        rem_slot_v = []
        for slot, values in target_kb.items():
            for v in values:
                if random.random() < rand_p:
                    rem_slot_v.append((slot, v))

        for slot_v in rem_slot_v:
            slot, v = slot_v
            v_idx = target_kb[slot].index(v)
            del target_kb[slot][v_idx]

            if len(target_kb[slot]) == 0:
                del target_kb[slot]

    def init_dialog(self, *args):
        """
        run when dialog starting
        :param args: tar_kb, tar_name
        :return:
        """
        self.target_kb = copy.deepcopy(args[0])
        self.target_name = args[1]

        SimulatorAct.random_mask_(self.target_kb, self.rand_p)

    def select_value(self, agent_act):
        user_value = self.target_kb[agent_act]

        # random select
        select_idx = random.randrange(0, len(user_value))

        user_value = user_value[select_idx]
        del self.target_kb[agent_act][select_idx]

        if len(self.target_kb[agent_act]) == 0:
            del self.target_kb[agent_act]

        return user_value

    def respond_act(self, agent_act, agent_value):
        """
        Respond to agent question
        :param agent_act: AgentActs type. (agent_act is none when dialog starting)
        :return: user_value, user_nl
        """
        # select_flag = False
        if agent_act is None:
            # select_flag = True
            # random select one kb to start dialog
            agent_act = random.choice(list(self.target_kb.keys()))

        # guess action
        if agent_act == AgentActs.GUESS:
            # only for user response. dialog is success when user target is in top_R results
            user_value = self.target_name == agent_value

        # unknown
        elif agent_act not in self.target_kb:
            user_value = None

        # have the kb
        else:
            # user_value = None

            # random mask
            # if select_flag or random.random() > self.rand_p:

            # answer partial kb
            if not self.enable_full_answer:
                user_value = self.select_value(agent_act)

            # answer full kb
            else:
                user_value = self.target_kb[agent_act]
                user_value = ', '.join(user_value)
                del self.target_kb[agent_act]

        user_nl = self.nl_template.act_to_nl(agent_act, user_value)

        return agent_act, user_value, user_nl

    def respond_nl(self, agent_nl):
        return NotImplementedError
