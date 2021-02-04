#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = "Han"
__email__ = "liuhan132@foxmail.com"


import random
import json


class AgentActs:
    """
    Pre-defined agent actions and slots
    """
    DIRECTED_BY = 'directed_by'
    RELEASE_YEAR = 'release_year'
    WRITTEN_BY = 'written_by'
    STARRED_ACTORS = 'starred_actors'
    HAS_GENRE = 'has_genre'
    HAS_TAGS = 'has_tags'
    IN_LANGUAGE = 'in_language'
    GUESS = 'guess'

    ALL_SLOTS = ['directed_by', 'release_year', 'written_by', 'starred_actors', 'has_genre', 'in_language']
    ALL_ACTIONS = ALL_SLOTS + ['guess']

    @staticmethod
    def slot_size():
        return len(AgentActs.ALL_SLOTS)

    @staticmethod
    def act_size():
        return len(AgentActs.ALL_ACTIONS)

    @staticmethod
    def contains_act(a):
        return a in AgentActs.ALL_ACTIONS

    @staticmethod
    def contains_slot(s):
        return s in AgentActs.ALL_SLOTS

    @staticmethod
    def slot_to_id(s):
        assert s in AgentActs.ALL_SLOTS, '%s not a valid slot' % s

        return AgentActs.ALL_SLOTS.index(s)

    @staticmethod
    def id_to_slot(idx):
        assert idx < len(AgentActs.ALL_SLOTS)

        return AgentActs.ALL_SLOTS[idx]

    @staticmethod
    def action_to_id(a):
        assert a in AgentActs.ALL_ACTIONS, '%s not a valid action' % a

        return AgentActs.ALL_ACTIONS.index(a)

    @staticmethod
    def id_to_action(idx):
        assert idx < len(AgentActs.ALL_ACTIONS)

        return AgentActs.ALL_ACTIONS[idx]


class NLTemplate:

    NO_ANSWER = ['I don`t know the answer.', 'I`m not sure about that.']

    def __init__(self, template_path):
        self.template_path = template_path

        with open(self.template_path, 'r') as f:
            self.nl_template = json.load(f)

    def act_to_nl(self, act_type, act_value, is_first=True):
        """
        action to natural language by filling the template
        :param act_type:
        :param act_value:
        :param is_first: whether the first time to ask this action in current dialog
        :return:
        """
        if AgentActs.contains_act(act_type):
            act_nl_temp = self.nl_template[act_type]
            nl_temp = act_nl_temp['nl_first'] if is_first else act_nl_temp['nl_more']

            if len(nl_temp) == 0:
                raise ValueError('No natural language template with action %s' % act_type)

            act_nl = random.choice(nl_temp)

            if len(act_nl_temp['slots']) > 0:
                assert len(act_nl_temp['slots']) == 1

                if act_value is None:
                    act_nl = random.choice(self.NO_ANSWER)
                else:
                    act_nl = act_nl.replace('$%s$' % act_nl_temp['slots'][0], str(act_value))

            return act_nl
        else:
            raise ValueError('Wrong value of act_type: %s' % str(act_type))
