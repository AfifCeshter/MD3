#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = "Han"
__email__ = "liuhan132@foxmail.com"

import logging
import torch
from game.tokenizer import Tokenizer, Vocabulary
from utils.functions import to_long_tensor, pad_sequences
from .doc_dataset import DocDataset

logger = logging.getLogger(__name__)


class BaseReader:
    """
    Base Dataset Reader
    """

    def __init__(self, game_config):
        self.dataset_config = game_config['dataset']
        self.data_prefix = self.dataset_config['data_prefix']

        self._max_doc_length = game_config['global']['max_doc_length']
        self._max_sent_length = game_config['global']['max_sent_length']
        self._max_sent_num = game_config['global']['max_sent_num']
        self._max_qa_length = game_config['global']['max_qa_length']
        self._max_turns = game_config['global']['max_turns']

        # read vocabulary
        self.vocab = Vocabulary(self.dataset_config['vocab_path'])

        # read documents data
        doc_id_path = self.dataset_config['doc_id_path']
        self.doc_reader = DocDataset(doc_id_path)

    def turn_to_tensor(self, agent_nl, user_nl):
        """
        convert the natural language of QA pairs to tensor
        :param agent_nl:
        :param user_nl:
        :return:
        """
        if agent_nl is None or user_nl is None:
            qa = Vocabulary.PAD
        else:
            qa = agent_nl + ' ' + user_nl

        return self.qa_to_tensor(qa)

    def qa_to_tensor(self, qa, padding=True):
        qa_split = Tokenizer.lower_tokenize(qa)
        qa_id = self.vocab.sentence_to_id(qa_split)

        # padding
        qa_id = qa_id[:self._max_qa_length] if len(qa_id) > self._max_qa_length else qa_id
        if padding:
            qa_id = qa_id + [0 for _ in range(self._max_qa_length - len(qa_id))]

        qa_tensor = to_long_tensor(qa_id)
        return qa_tensor

    def doc_to_tensor(self, cand_names, get_doc_id=None, hierarchical=False):
        """
        convert the documents on natural language to tensor
        :param get_doc_id: function for getting documents id by names
        :param hierarchical:
        :param cand_names:
        :return:
        """
        if get_doc_id is None:
            get_doc_id = self.doc_reader.get_doc_id

        if hierarchical:
            cand_docs_id_tensor = []
            for name in cand_names:

                # padding sentences
                cur_doc_id = get_doc_id(name, hierarchical)
                cur_doc_id_array = pad_sequences(cur_doc_id,
                                                 maxlen=self._max_sent_length,
                                                 padding='post',
                                                 value=Vocabulary.PAD_IDX)
                cur_doc_id_tensor = to_long_tensor(cur_doc_id_array)

                # padding to sentences number
                if cur_doc_id_tensor.size()[0] > self._max_sent_num:
                    cur_doc_id_tensor_pad = cur_doc_id_tensor[:self._max_sent_num, :]
                else:
                    padding_tensor = torch.zeros(self._max_sent_num - cur_doc_id_tensor.shape[0],
                                                 self._max_sent_length).long()
                    cur_doc_id_tensor_pad = torch.cat([cur_doc_id_tensor, padding_tensor], dim=0)

                cand_docs_id_tensor.append(cur_doc_id_tensor_pad)

            cand_docs_id_tensor = torch.stack(cand_docs_id_tensor, dim=0)

        else:
            cand_docs_id = list(map(lambda x: get_doc_id(x, hierarchical), cand_names))
            cand_docs_id_array = pad_sequences(cand_docs_id,
                                               maxlen=self._max_doc_length,
                                               padding='post',
                                               value=Vocabulary.PAD_IDX)
            cand_docs_id_tensor = to_long_tensor(cand_docs_id_array)

        return cand_docs_id_tensor


class SLReader(BaseReader):
    """
    Supervised learning reader
    """

    def __init__(self, game_config):
        super(SLReader, self).__init__(game_config)

        self._data_train_path = self.dataset_config['data_train_path']
        self._data_dev_path = self.dataset_config['data_dev_path']
        self._data_test_path = self.dataset_config['data_test_path']

    def _get_dataloader(self, data_path, batch_size, num_workers, shuffle):
        return NotImplementedError

    def get_dataloader_train(self, batch_size, num_workers):
        """
        Train dialog data loader for supervised training
        :param batch_size:
        :param num_workers:
        :return:
        """
        return self._get_dataloader(self._data_train_path, batch_size, num_workers, shuffle=True)

    def get_dataloader_dev(self, batch_size, num_workers):
        """
        Dev dialog data loader for supervised training
        :param batch_size:
        :param num_workers:
        :return:
        """
        return self._get_dataloader(self._data_dev_path, batch_size, num_workers, shuffle=False)

    def get_dataloader_test(self, batch_size, num_workers):
        """
        Test dialog data loader for supervised training
        :param batch_size:
        :param num_workers:
        :return:
        """
        return self._get_dataloader(self._data_test_path, batch_size, num_workers, shuffle=False)
