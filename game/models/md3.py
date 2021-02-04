#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = "Han"
__email__ = "liuhan132@foxmail.com"

"""MD3 model with NLU, DST and Policy"""

import logging
import numpy as np
import torch.nn
import torch.nn.functional as F
from game.models.layers import *
from utils.functions import compute_mask, del_zeros_right
from game.tokenizer import Vocabulary
from game.template import AgentActs

logger = logging.getLogger(__name__)


class PTNLU(torch.nn.Module):
    """
    StateNet: rank the documents and filter

    Args:
        model_config: config
        embedding_path:

    Inputs:
        docs_rep: (num_docs, num_slots, hidden_size * 4)
            - every sample have the same documents representation
        turn_nl: (batch, turn_len)

    Outputs:
        docs_dist_update: (batch, num_docs)
        turn_slot_cls: (batch, num_slots + 1)
    """

    def __init__(self, model_config, embedding_path, embedding_freeze=True):
        super(PTNLU, self).__init__()

        self.model_config = model_config

        embedding_num = model_config['embedding_num']
        embedding_dim = model_config['embedding_dim']
        self.hidden_size = model_config['hidden_size']
        dropout_p = model_config['dropout_p']
        enable_layer_norm = model_config['layer_norm']
        num_slots = AgentActs.slot_size()

        if not model_config['use_glove']:
            self.embedding_layer = torch.nn.Embedding(num_embeddings=embedding_num,
                                                      embedding_dim=embedding_dim,
                                                      padding_idx=Vocabulary.PAD_IDX)
        else:
            embedding_weight = torch.tensor(np.load(embedding_path), dtype=torch.float32)
            logger.info('Embedding shape: ' + str(embedding_weight.shape))
            self.embedding_layer = torch.nn.Embedding.from_pretrained(embedding_weight,
                                                                      freeze=embedding_freeze,
                                                                      padding_idx=Vocabulary.PAD_IDX)

        self.turn_rnn = MyRNNBase(mode='GRU',
                                  input_size=embedding_dim,
                                  hidden_size=self.hidden_size,
                                  bidirectional=True,
                                  dropout_p=dropout_p,
                                  enable_layer_norm=enable_layer_norm,
                                  batch_first=True)
        self.turn_cls_linear = torch.nn.Linear(self.hidden_size * 2, num_slots)
        self.turn_inform_linear = torch.nn.Linear(self.hidden_size * 2, 1)
        self.similar_bilinear = torch.nn.Bilinear(self.hidden_size * 4, self.hidden_size * 2, 1)

    def forward(self, docs_rep, turn_nl):
        turn_nl, _ = del_zeros_right(turn_nl)

        # 1. user response encoding
        turn_emb = self.embedding_layer(turn_nl)
        turn_mask = compute_mask(turn_nl)

        # (batch, hidden_size * 2)
        turn_word_rep, turn_rep = self.turn_rnn(turn_emb, turn_mask)

        # 1.1 user response classification
        # (batch, num_slots)
        turn_slot_cls = self.turn_cls_linear(turn_rep)
        turn_slot_cls = torch.softmax(turn_slot_cls, dim=-1)

        # (batch, 1)
        turn_inform_sig = self.turn_inform_linear(turn_rep)
        turn_inform_sig = torch.sigmoid(turn_inform_sig)

        # (batch, num_slots + 1)
        turn_slot_inform_cls = torch.cat([(1 - turn_inform_sig) * turn_slot_cls,
                                          turn_inform_sig], dim=-1)

        # 2. compute similar with bi-linear layer
        batch, num_docs, num_slots, _ = docs_rep.size()

        # (batch, num_docs, num_slots, hidden_size * 2)
        turn_rep_expand = turn_rep.unsqueeze(1).unsqueeze(1).expand(-1, num_docs, num_slots, -1).contiguous()

        # (batch, num_docs, num_slots)
        turn_docs_slots_similar = self.similar_bilinear(docs_rep, turn_rep_expand).squeeze(-1)
        extra_similar = turn_docs_slots_similar.new_ones(batch, num_docs, 1)

        # 2.1 similar with slots weighted
        # (batch, num_docs, num_slots + 1)
        turn_docs_slots_similar_extra = torch.cat([turn_docs_slots_similar, extra_similar], dim=-1)
        # (batch * num_docs, num_slots + 1)
        turn_docs_slots_similar_extra = turn_docs_slots_similar_extra.reshape(batch * num_docs, -1)

        # (batch, num_docs, num_slots + 1)
        turn_slot_inform_cls_expand = turn_slot_inform_cls.unsqueeze(1).expand(-1, num_docs, -1)
        # (batch * num_docs, num_slots + 1)
        turn_slot_inform_cls_expand = turn_slot_inform_cls_expand.reshape(batch * num_docs, -1)

        # (batch * num_docs)
        turn_docs_similar = torch.bmm(turn_docs_slots_similar_extra.unsqueeze(1),
                                      turn_slot_inform_cls_expand.unsqueeze(2)).squeeze(1).squeeze(1)
        # (batch, num_docs)
        turn_level_docs_dist = torch.softmax(turn_docs_similar.view(batch, num_docs), dim=-1)

        return turn_level_docs_dist, turn_slot_cls, turn_inform_sig


class PTDST(torch.nn.Module):
    """
    DST Module: GRU cell

    Args:
        model_config: config

    Inputs:
        turn_rep: (batch, hidden_size * 2)

    Outputs:
        dialog_rep: (batch, hidden_size)
    """

    def __init__(self, model_config):
        super(PTDST, self).__init__()

        hidden_size = model_config['hidden_size']
        self.dialog_gru = torch.nn.GRUCell(input_size=hidden_size * 2,
                                           hidden_size=hidden_size)

    def forward(self, turn_rep, last_dialog_rep):
        dialog_rep = self.dialog_gru(turn_rep, last_dialog_rep)
        return dialog_rep


class PTPolicyNet(torch.nn.Module):
    """
    PolicyNet: select one action to query

    Args:
        model_config: config

    Inputs:
        docs_diff_rep: (batch, num_docs, num_slots, hidden_size * 4)
        docs_prob: (batch, num_docs)
        dialog_his: (batch, hidden_size)

    Outputs:
        acts_prob: (batch, num_slots)
    """

    def __init__(self, model_config):
        super(PTPolicyNet, self).__init__()

        self.model_config = model_config
        self.hidden_size = model_config['hidden_size']
        dropout_p = model_config['dropout_p']

        self.act_linear = torch.nn.Linear(self.hidden_size * 4, 1)

    def forward(self, docs_diff_rep, dialog_level_doc_dist, dialog_level_act_dist):
        # docs_mean_rep = docs_rep.mean(dim=1)  # (batch, num_slots, hidden_size * 4)
        # docs_diff_rep = (docs_rep - docs_mean_rep.unsqueeze(1)).pow(2)

        # 1. weighted by documents distribution
        batch, num_docs, num_slots, _ = docs_diff_rep.size()
        docs_diff_rep = docs_diff_rep.view(batch, num_docs, -1)

        # (batch, num_slots * hidden_size * 4)
        slots_diff_rep = torch.bmm(docs_diff_rep.transpose(1, 2),
                                   dialog_level_doc_dist.unsqueeze(-1)).squeeze(-1)
        # (batch, num_slots, hidden_size * 4)
        slots_diff_rep = slots_diff_rep.view(batch, num_slots, -1)

        # 2. generate docs-diff action distribution
        # (batch, num_slots)
        acts_prob = self.act_linear(slots_diff_rep).squeeze(-1)
        # acts_prob = torch.softmax(acts_prob, dim=-1)

        # 3. mix with dialog history
        acts_prob_with_his = acts_prob * (1 - dialog_level_act_dist)
        acts_prob_with_his = torch.softmax(acts_prob_with_his, dim=-1)
        # acts_prob_with_his = F.normalize(acts_prob_with_his, p=2, dim=-1)
        # acts_prob_with_his += 1e-7

        # acts_prob_with_his = acts_prob  # for no-dst used

        return acts_prob_with_his
