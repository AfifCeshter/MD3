#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = "Han"
__email__ = "liuhan132@foxmail.com"

# dialog framework that managing agent and simulator

import torch
import json
import logging
from tqdm import tqdm
from collections import deque
from game.simulator import SimulatorAct
import game.agent
import game.dataset_reader
from game.template import AgentActs
from utils.functions import get_entropy

# DialogStatus = namedtuple('DialogStatus', ('reward', 'turn', 'is_success', 'is_end'))

# Define the dialog state for input
# DialogState = namedtuple('DialogState', ('last_docs_prob', 'last_acts_prob', 'last_turn'))

logger = logging.getLogger(__name__)


class MultiDM:
    """
    Manage the dialog agent and dialog simulator
    """

    def __init__(self, game_config, writer, device):
        self.game_config = game_config

        self.max_turns = game_config['global']['max_turns']
        self.max_qa_len = game_config['global']['max_qa_length']
        self.max_doc_len = game_config['global']['max_doc_length']

        self.enable_full_answer = game_config['global']['full_answer']

        self.num_workers = game_config['global']['num_data_workers']
        self.dialog_data_path = game_config['checkpoint']['dialog_data_path']
        self.false_dialog_data_path = game_config['checkpoint']['false_dialog_data_path']

        # summary writer
        self.writer = writer

        # device
        self.device = device

        # for agent
        self.agent_type = game_config['global']['agent_type']
        self.agent = self.get_agent()

        # for simulator
        self.simulator_type = game_config['global']['simulator_type']
        self.simulator = self.get_simulator()

        # dataset:
        self.dataset = self.get_dataset()

        # train config
        self.num_workers = game_config['global']['num_data_workers']
        self.batch_size = game_config['train']['batch_size']
        self.evaluate_freq = game_config['train']['evaluate_freq']
        self.evaluate_steps = game_config['train']['evaluate_steps']
        self.save_steps = game_config['train']['save_steps']
        self.train_iters = game_config['train']['train_iters']
        self.test_iters = game_config['train']['test_iters']

        self.num_steps = 0
        self.training = False
        self.init_deque()

    def get_agent(self):
        """
        Get different agent
        """
        name_class = {
            'rule-none': 'AgentRulesNone',
            'rule-rand': 'AgentRulesRand',
            'rule-fixed': 'AgentRulesFixed',
            'rule-rule': 'AgentRulesRules',
            'model-none': 'AgentModelNone',
            'model-rand': 'AgentModelRand',
            'model-fixed': 'AgentModelFixed',
            'model-rule': 'AgentModelRules',
            'model-model': 'AgentModelModel',
            'mrc-rand': 'AgentMRCRand',
            'mrc-fixed': 'AgentMRCFixed',
            'mrc-model': 'AgentMRCModel'
        }
        if self.agent_type in name_class.keys():
            return getattr(game.agent, name_class[self.agent_type])(self.game_config, self.device)
        raise ValueError('%s is not a valid value for agent type' % self.agent_type)

    def get_simulator(self):
        """
        Get different simulator
        """
        if self.simulator_type == 'rule':
            return SimulatorAct(self.game_config['dataset']['simulator_template_path'],
                                self.enable_full_answer,
                                self.game_config['global']['simulator_rand'])
        elif self.simulator_type == 'model':
            raise ValueError('Not implemented')

        raise ValueError('%s is not a valid value for simulator type' % self.simulator_type)

    def get_dataset(self):
        """
        Get different dataset
        """
        name_class = {
            'rule-none': 'CandKBKBReader',
            'rule-rand': 'CandKBKBReader',
            'rule-fixed': 'CandKBKBReader',
            'rule-rule': 'CandKBKBReader',
            'model-none': 'CandDocDocReader',
            'model-rand': 'CandDocDocReader',
            'model-fixed': 'CandDocDocReader',
            'model-rule': 'CandDocKBReader',
            'model-model': 'CandDocDocReader',
            'mrc-rand': 'CandMRCMRCReader',
            'mrc-fixed': 'CandMRCMRCReader',
            'mrc-model': 'CandMRCDocReader'
        }
        if self.agent_type in name_class.keys():
            return getattr(game.dataset_reader, name_class[self.agent_type])(self.game_config)
        raise ValueError('%s is not a valid value for agent type' % self.agent_type)

    def init_deque(self):
        """
        Initial metrics deque on training and testing
        """
        if self.training:
            self.top_1_success = deque(maxlen=self.evaluate_steps)
            self.top_3_success = deque(maxlen=self.evaluate_steps)
            self.mrr = deque(maxlen=self.evaluate_steps)
            self.num_turns = deque(maxlen=self.evaluate_steps)
            self.rewards = deque(maxlen=self.evaluate_steps)
        else:
            self.top_1_success = deque()
            self.top_3_success = deque()
            self.mrr = deque()
            self.num_turns = deque()
            self.rewards = deque()

    def get_metrics(self):
        assert len(self.top_1_success) == len(self.top_3_success) and len(self.top_3_success) == len(self.num_turns) \
               and len(self.num_turns) == len(self.rewards) and len(self.rewards) > 0

        ave_top1_success = sum(self.top_1_success) / len(self.top_1_success)
        ave_top3_success = sum(self.top_3_success) / len(self.top_3_success)
        ave_mrr = sum(self.mrr) / len(self.mrr)
        ave_turns = sum(self.num_turns) / len(self.num_turns)
        ave_rewards = sum(self.rewards) / len(self.rewards)

        return ave_top1_success, ave_top3_success, ave_mrr, ave_turns, ave_rewards

    def test_dataset(self):
        """
        Testing on all dataset
        :return:
        """
        datareader = self.dataset.get_dataset_reader(num_workers=self.num_workers,
                                                     iters=self.test_iters)
        # datareader = self.dataset.get_fixed_reader(num_workers=self.num_workers)  # config['dataset']['cand_doc_path']

        self.training = False
        self.agent.eval_config()
        self.init_deque()

        dialog_data = []
        false_dialog_data = []

        # test on fixed runs
        for example in tqdm(datareader, desc='Testing game...'):
            # simulate a dialog
            cur_dialog_data, cur_top_1_success, cur_top_3_success, cur_mrr, cur_num_turns, cur_rewards = \
                self.run_dialog(*example)
            dialog_data.append(cur_dialog_data)

            self.top_1_success.append(cur_top_1_success)
            self.top_3_success.append(cur_top_3_success)
            self.mrr.append(cur_mrr)
            self.num_turns.append(cur_num_turns)
            self.rewards.append(cur_rewards)

            if not cur_top_1_success:
                false_dialog_data.append(cur_dialog_data)

        # save the dialog history
        self.save_dialog(dialog_data, self.dialog_data_path)
        self.save_dialog(false_dialog_data, self.false_dialog_data_path)

        # metrics
        return self.get_metrics()

    def train_dataset(self):
        """
        Training the Dialog Model
        :return:
        """
        assert 'rule' not in self.agent_type

        self.training = True
        self.agent.train_config()
        self.init_deque()

        datareader = self.dataset.get_dataset_reader(num_workers=self.num_workers,
                                                     iters=self.train_iters)

        for example in tqdm(datareader, desc='Training game...'):
            # simulate a dialog
            cur_dialog_data, cur_top_1_success, cur_top_3_success, cur_mrr, cur_num_turns, cur_rewards = \
                self.run_dialog(*example)

            self.top_1_success.append(cur_top_1_success)
            self.top_3_success.append(cur_top_3_success)
            self.mrr.append(cur_mrr)
            self.num_turns.append(cur_num_turns)
            self.rewards.append(cur_rewards)

        # save model
        self.agent.save_parameters(self.num_steps)

    def run_dialog(self, cand_docs, cand_docs_diff, cand_names, tar_kb, tar_name):
        """
        Simulate a dialog with model
        :param cand_docs:
        :param cand_names:
        :param tar_kb:
        :param tar_name:
        :return:
        """
        self.agent.init_dialog(cand_docs, cand_docs_diff, cand_names)
        self.simulator.init_dialog(tar_kb, tar_name)

        tar_idx = cand_names.index(tar_name)

        # dialog status
        top_1_success = False
        top_3_success = False
        mrr = 0
        all_rewards = []
        saved_log_act_probs = []  # saved action log probability for REINFORCE algorithm
        saved_log_doc_probs = []  # saved docs log probability for REINFORCE algorithm
        num_turns = self.max_turns

        # dialog start with user
        # user_act, user_value, user_nl = self.simulator.respond_act(agent_act=None, agent_value=None)
        # dialog_json = [{'turn_id': 0,
        #                 'agent_act': '',
        #                 'agent_value': '',
        #                 'agent_nl': '',
        #                 'user_act': user_act,
        #                 'user_value': user_value,
        #                 'user_nl': user_nl,
        #                 'tar_rank': 0,
        #                 'docs_entropy': 0}]

        # if 'rule-' in self.agent_type:
        #     last_turn = (user_act, user_value)
        # else:
        #     last_turn = self.dataset.turn_to_tensor(agent_nl='', user_nl=user_nl).unsqueeze(0)
        dialog_his = '_BOS_'
        dialog_json = []

        last_turn = None

        assert self.max_turns > 1
        for turn_i in range(1, self.max_turns + 1):
            agent_act, agent_act_prob, agent_value, agent_nl = self.agent.turn_act(last_turn)

            # feedback from environments
            if len(dialog_json) == 0:
                is_end = agent_act == AgentActs.GUESS
            else:
                reward, top_1_success, top_3_success, is_end, mrr, tar_r, docs_entropy, tar_prob = \
                    self.env_feedback(agent_act,
                                      self.agent.dialog_level_doc_dist,
                                      tar_idx)
                all_rewards.append(reward)
                dialog_json[-1]['tar_rank'] = tar_r
                dialog_json[-1]['tar_prob'] = float('{:.2f}'.format(tar_prob))
                dialog_json[-1]['docs_entropy'] = float('{:.2f}'.format(docs_entropy))

            # user response
            user_act, user_value, user_nl = self.simulator.respond_act(agent_act, agent_value)

            # record current turn
            if 'rule-' in self.agent_type or 'mrc' in self.agent_type:
                last_turn = (user_act, user_value)
            else:
                last_turn = self.dataset.turn_to_tensor(agent_nl=agent_nl, user_nl=user_nl).unsqueeze(0)

            # record data
            turn_json = {'turn_id': turn_i,
                         'agent_act': agent_act,
                         'agent_value': agent_value,
                         'agent_nl': agent_nl,
                         'user_act': user_act,
                         'user_value': user_value,
                         'user_nl': user_nl,
                         'tar_rank': 0,
                         'tar_prob': 0,
                         'docs_entropy': 0,
                         'docs_dist': self.agent.dialog_level_doc_dist.tolist()}
            dialog_json.append(turn_json)

            # early stop
            if is_end:
                num_turns = turn_i
                break

            # saved log probability for rl-training
            if self.training:
                agent_act_id = AgentActs.slot_to_id(agent_act)
                saved_log_act_probs.append(torch.log(agent_act_prob[agent_act_id]))

                pred_doc_id = cand_names.index(agent_value)
                saved_log_doc_probs.append(torch.log(self.agent.dialog_level_doc_dist[0][pred_doc_id]))

        # steps on episode
        self.num_steps += 1

        # when training
        if self.training and len(all_rewards):
            loss = self.agent.update(all_rewards, saved_log_act_probs, saved_log_doc_probs)  # update the model
            self.on_training(loss)

        cur_rewards = sum(all_rewards)
        data_json = {'tar_name': tar_name,
                     'cand_names': cand_names,
                     'dialog': dialog_json,
                     'top_1_success': top_1_success,
                     'top_3_success': top_3_success,
                     'mrr': mrr,
                     'rewards': '{:.2f}'.format(cur_rewards)}

        return data_json, top_1_success, top_3_success, mrr, num_turns, cur_rewards

    def on_training(self, loss):
        """
        Training for REINFORCE algorithm
        """
        self.writer.add_scalar('Train-Loss', loss, global_step=self.num_steps)

        # save model parameters
        if self.num_steps % self.save_steps == 0:
            logger.debug('step=%d/%d: saving model parameters...' % (self.num_steps, self.train_iters))
            self.agent.save_parameters(self.num_steps)

        # show messages when training
        if self.num_steps % self.evaluate_freq == 0:
            ave_top1_success, ave_top3_success, ave_mrr, ave_turns, ave_rewards = self.get_metrics()
            logger.debug('steps=%d/%d, ave_top1_success=%.2f, ave_top3_success=%.2f, ave_turns=%.2f, ave_rewards=%.2f'
                         % (self.num_steps, self.train_iters, ave_top1_success, ave_top3_success, ave_turns, ave_rewards))

            self.writer.add_scalar('Train-Ave-Top1-Success', ave_top1_success, global_step=self.num_steps)
            self.writer.add_scalar('Train-Ave-Top3-Success', ave_top3_success, global_step=self.num_steps)
            self.writer.add_scalar('Train-Ave-MRR', ave_mrr, global_step=self.num_steps)
            self.writer.add_scalar('Train-Ave-Turns', ave_turns, global_step=self.num_steps)
            self.writer.add_scalar('Train-Ave-Rewards', ave_rewards, global_step=self.num_steps)

    def save_dialog(self, dialog_data, save_path):
        """
        save dialog to json file
        :param dialog_data:
        :param save_path:
        :return:
        """
        with open(save_path, 'w') as wf:
            json.dump(dialog_data, wf, indent=2)

    def reward_function(self, tar_r, top_R, is_success, is_end):
        """
        reward function for reinforcement learning
        :param tar_r:
        :param is_success:
        :param is_end:
        :return:
        """
        if is_end:
            if is_success:
                reward = max(0, 2 * (1 - (tar_r - 1) / top_R))
            else:
                reward = -1
        else:
            reward = -0.1
        return reward

    def env_feedback(self, agent_act, docs_prob, tar_idx):
        """
        Get feedback from environments: (reward, is_success and is_end)
        """
        top_1_success = False
        top_3_success = False
        top_5_success = False
        is_end = False

        # guess action
        docs_prob_sort, sort_idx = torch.sort(docs_prob, dim=-1, descending=True)
        tar_r = sort_idx.squeeze(0).cpu().data.tolist().index(tar_idx) + 1
        tar_prob = docs_prob_sort[0][tar_r-1].cpu().item()
        mrr = 1 / tar_r

        if agent_act == AgentActs.GUESS:
            # is_success = user_value
            top_1_success = tar_r <= 1  # dialog is success if the user target is in top_R results
            top_3_success = tar_r <= 3
            top_5_success = tar_r <= 5
            is_end = True

        # reward function
        reward = self.reward_function(tar_r, 5, top_5_success, is_end)

        # docs probability entropy
        docs_entropy = get_entropy(docs_prob)

        return reward, top_1_success, top_3_success, is_end, mrr, tar_r, docs_entropy, tar_prob
