#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = "Han"
__email__ = "liuhan132@foxmail.com"

"""Documents knowledge dataset"""

import os
import json
from game.tokenizer import Vocabulary
from utils.functions import list_chunk
from collections import OrderedDict


class DocDataset:
    """
    Documents knowledge reader
    """

    def __init__(self, doc_id_path):
        self.doc_id_path = doc_id_path

        # read documents data
        self.doc_data = self._read_doc()

    def get_all_names(self):
        return list(self.doc_data.keys())

    def get_all_kb(self):
        return list(map(lambda x: x['kb'], self.doc_data.values()))

    def get_kb(self, name, is_full):
        name = name.lower()
        if name in self.doc_data:
            if is_full:
                return self.doc_data[name]['kb_full']
            return self.doc_data[name]['kb']
        raise ValueError(name + 'not exists in documents data')

    def get_doc(self, name, hierarchical=False):
        name_low = name.lower()
        if name_low in self.doc_data:
            cur_doc = self.doc_data[name_low]['doc']
            if not hierarchical:
                return cur_doc
            else:
                cur_doc_hc = list_chunk(cur_doc, Vocabulary.SEP)
                return cur_doc_hc

        raise ValueError('"%s" not exists in documents data' % name_low)

    def get_doc_id(self, name, hierarchical):
        name_low = name.lower()

        if name_low in self.doc_data:
            cur_doc_id = self.doc_data[name_low]['doc_id']

            if not hierarchical:
                return cur_doc_id
            else:
                cur_doc_id_hc = list_chunk(cur_doc_id, Vocabulary.SEP_IDX)
                return cur_doc_id_hc

        raise ValueError('"%s" not exists in documents data' % name_low)

    def _read_doc(self):
        assert os.path.exists(self.doc_id_path)

        with open(self.doc_id_path, 'r') as f:
            doc_data = json.load(f)

        doc_dict = OrderedDict()    # order needed
        for ele in doc_data:
            doc_dict[ele['name']] = ele

        return doc_dict
