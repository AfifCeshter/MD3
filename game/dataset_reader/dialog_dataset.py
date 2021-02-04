#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = "Han"
__email__ = "liuhan132@foxmail.com"


import logging
import json
import torch
from torch.utils.data.sampler import Sampler
from game.template import AgentActs
from .base_reader import BaseReader
from .doc_dataset import DocDataset
from .doc_rep_pt_dataset import load_docs_rep

logger = logging.getLogger(__name__)


class DialogReader(BaseReader):
    """
    Dialog dataset reader with turns
    """

    def __init__(self, game_config):
        super(DialogReader, self).__init__(game_config)

    def _get_dataloader(self, batch_size, num_workers, shuffle, dialog_path, data_path, docs_rep_path_prefix):
        """
        Dialog data loader for supervised training
        :param dialog_path:
        :param batch_size:
        :param num_workers:
        :param shuffle:
        :return:
        """
        dialog_dataset = DialogTurnsDataset(dialog_path,
                                            data_path,
                                            docs_rep_path_prefix,
                                            self.qa_to_tensor,
                                            self._max_qa_length)
        return torch.utils.data.DataLoader(dialog_dataset,
                                           batch_size=batch_size,
                                           collate_fn=DialogTurnsDataset.collect_fun,
                                           num_workers=num_workers,
                                           shuffle=shuffle)

    def get_dataloader_train(self, batch_size, num_workers):
        return self._get_dataloader(batch_size,
                                    num_workers=num_workers,
                                    shuffle=True,
                                    dialog_path='outputs/20200319/rep-dialog-kb-50k/dialog_data.json',
                                    data_path=self.data_prefix + 'doc_rep/guessmovie_rep_doc_id.json',
                                    docs_rep_path_prefix=self.data_prefix + 'doc_rep/pt/rep_doc_pt_rep.pt')

    def get_dataloader_test(self, batch_size, num_workers):
        return self._get_dataloader(batch_size,
                                    num_workers=num_workers,
                                    shuffle=True,
                                    dialog_path='outputs/20200318/dialog-dialog-kb/dialog_data.json',
                                    data_path=self.data_prefix + 'guessmovie_dialog_doc_id.json',
                                    docs_rep_path_prefix=self.data_prefix + 'doc_rep/pt/dialog_doc_pt_rep.pt')


class DialogTurnsDataset(torch.utils.data.Dataset):
    """
    Dialog on turns dataset
    """

    def __init__(self, dialog_path, data_path, docs_rep_path_prefix, qa_to_tensor, max_qa_len):
        super(DialogTurnsDataset, self).__init__()

        self.max_qa_len = max_qa_len

        # build documents dataset reader
        self.doc_reader = DocDataset(data_path)
        self.all_docs_names = self.doc_reader.get_all_names()

        self.docs_rep = load_docs_rep(docs_rep_path_prefix)
        self.qa_to_tensor = qa_to_tensor

        with open(dialog_path, 'r') as f:
            dialog_data = json.load(f)

        self.turn_data = self.dialog_to_turns(dialog_data)

    def __len__(self):
        return len(self.turn_data)

    def __getitem__(self, index):
        turn = self.turn_data[index]

        tar_name = turn['tar_name']
        cand_docs_names = turn['cand_names']

        cand_docs_idx = [self.all_docs_names.index(name) for name in cand_docs_names]
        cand_doc_tensor = self.docs_rep[cand_docs_idx, :]

        # get the dialog history tensor
        dia_his_tensor = self.qa_to_tensor(turn['dialog_history']).long()

        # get the ground truth of agent act and value
        agent_act = AgentActs.slot_to_id(turn['agent_act'])

        # target documents and documents distribution
        tar_idx = cand_docs_names.index(tar_name)
        docs_dist = torch.tensor(turn['docs_dist'], dtype=torch.float).gt(0).long().squeeze(0)

        return cand_doc_tensor, dia_his_tensor, agent_act, tar_idx, docs_dist

    def dialog_to_turns(self, dialog_data):
        """
        convert dialog data to turn data using for supervised training directly
        :return:
        """
        turn_data = []

        for dialog in dialog_data:
            if not dialog['top_1_success']:
                logger.debug('Ignore the failed dialog with tar_name=%s' % dialog['tar_name'])

            cand_names = dialog['cand_names']
            tar_name = dialog['tar_name']
            dialog_history = "_BOS_"
            for turn in dialog['dialog'][:-1]:
                turn_data.append({'cand_names': cand_names,
                                  'tar_name': tar_name,
                                  'dialog_history': dialog_history,
                                  'agent_act': turn['agent_act'],
                                  'docs_dist': turn['docs_dist']})

                dialog_history += ' ' + turn['agent_nl'] + ' ' + turn['user_nl']
        return turn_data

    @staticmethod
    def collect_fun(batch):
        """
        collect funtion for batch
        :param batch:
        :return:
        """
        batch_cand_doc = []
        batch_dia_his = []
        batch_agent_act = []
        batch_tar_doc = []
        batch_docs_dist = []

        for ele in batch:
            batch_cand_doc.append(ele[0])
            batch_dia_his.append(ele[1])
            batch_agent_act.append(ele[2])
            batch_tar_doc.append(ele[3])
            batch_docs_dist.append(ele[4])

        batch_cand_doc_tensor = torch.stack(batch_cand_doc, dim=0)
        batch_dia_his_tensor = torch.stack(batch_dia_his, dim=0)
        batch_agent_act_tensor = torch.tensor(batch_agent_act, dtype=torch.long)
        batch_tar_doc_tensor = torch.tensor(batch_tar_doc, dtype=torch.long)
        batch_docs_dist_tensor = torch.stack(batch_docs_dist, dim=0)

        return batch_cand_doc_tensor, batch_dia_his_tensor, batch_agent_act_tensor, \
               batch_tar_doc_tensor, batch_docs_dist_tensor
