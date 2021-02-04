#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = "Han"
__email__ = "liuhan132@foxmail.com"

import logging
import numpy as np
import torch.nn
import torch.cuda
from game.models.layers import *
from utils.functions import compute_mask, del_zeros_right, compute_top_layer_mask
from game.tokenizer import Vocabulary

logger = logging.getLogger(__name__)


class DocRepPTTrainModel(torch.nn.Module):
    """
    Documents representation model

    Args:
        model_config: config
        embedding_path: embeddings path

    Inputs:
        tar_d: (batch, doc_sent_len, doc_word_len)
        cand_d: (batch, cand_doc_num, doc_sent_len, doc_word_len)

    Outputs:
        cand_d_prop: (batch, cand_doc_num)
    """

    def __init__(self, model_config, embedding_path=None, embedding_freeze=True):
        super(DocRepPTTrainModel, self).__init__()

        self.model_config = model_config

        embedding_num = model_config['embedding_num']
        embedding_dim = model_config['embedding_dim']

        self.hidden_size = model_config['hidden_size']

        dropout_p = model_config['dropout_p']
        enable_layer_norm = model_config['layer_norm']
        self.doc_hierarchical = model_config['doc_hierarchical']

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

        self.tar_doc_encoder = DocRepPTEncoder(embedding_dim, self.hidden_size, dropout_p, enable_layer_norm)
        self.cand_doc_encoder = DocRepPTEncoder(embedding_dim, self.hidden_size, dropout_p, enable_layer_norm)
        # self.tar_doc_encoder = TransformerModel(nemb=embedding_dim,
        #                                         nhead=2,
        #                                         nhid=200,
        #                                         nlayers=2,
        #                                         dropout=dropout_p)
        # self.cand_doc_encoder = TransformerModel(nemb=embedding_dim,
        #                                          nhead=2,
        #                                          nhid=200,
        #                                          nlayers=2,
        #                                          dropout=dropout_p)

    def forward(self, tar_d, cand_ds):
        tar_d, _ = del_zeros_right(tar_d)
        cand_ds, _ = del_zeros_right(cand_ds)

        if self.doc_hierarchical:
            _, sent_right_idx = del_zeros_right(tar_d.sum(-1))
            tar_d = tar_d[:, :sent_right_idx, :]

            _, sent_right_idx = del_zeros_right(cand_ds.sum(-1))
            cand_ds = cand_ds[:, :, :sent_right_idx, :]

        # embedding layer
        tar_doc_emb = self.embedding_layer(tar_d)
        tar_doc_mask = compute_mask(tar_d)

        cand_docs_emb = self.embedding_layer(cand_ds)
        cand_docs_mask = compute_mask(cand_ds)

        # target document encoder layer
        tar_doc_rep, _ = self.tar_doc_encoder(tar_doc_emb, tar_doc_mask)

        # candidate documents encoder layer
        batch, cand_doc_num = cand_docs_emb.size(0), cand_docs_emb.size(1)
        new_size = [batch * cand_doc_num] + list(cand_docs_emb.shape[2:])
        cand_docs_emb_flip = cand_docs_emb.view(*new_size)

        new_size = [batch * cand_doc_num] + list(cand_ds.shape[2:])
        cand_docs_mask_flip = cand_docs_mask.view(*new_size)

        cand_docs_rep_flip, _ = self.cand_doc_encoder(cand_docs_emb_flip, cand_docs_mask_flip)
        cand_docs_rep = cand_docs_rep_flip.contiguous().view(batch, cand_doc_num, -1)

        # output layer
        cand_scores = torch.bmm(tar_doc_rep.unsqueeze(1),
                                cand_docs_rep.transpose(1, 2)).squeeze(1)  # (batch, cand_doc_num)
        cand_logits = torch.log_softmax(cand_scores, dim=-1)

        return cand_logits


class DocRepPTTestModel(torch.nn.Module):
    """
    Documents representation out model

    Args:
        model_config: config
        embedding_path: embeddings path

    Inputs:
        doc: (batch, doc_sent_len, doc_word_len)

    Outputs:
        document_rep: (batch, hidden_size * 4)
    """

    def __init__(self, model_config, embedding_path=None, embedding_freeze=True):
        super(DocRepPTTestModel, self).__init__()

        self.model_config = model_config

        embedding_num = model_config['embedding_num']
        embedding_dim = model_config['embedding_dim']

        self.hidden_size = model_config['hidden_size']

        dropout_p = model_config['dropout_p']
        enable_layer_norm = model_config['layer_norm']

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

        self.tar_doc_encoder = DocRepPTEncoder(embedding_dim, self.hidden_size, dropout_p, enable_layer_norm)
        self.cand_doc_encoder = DocRepPTEncoder(embedding_dim, self.hidden_size, dropout_p, enable_layer_norm)
        # self.tar_doc_encoder = TransformerModel(nemb=embedding_dim,
        #                                         nhead=2,
        #                                         nhid=200,
        #                                         nlayers=2,
        #                                         dropout=dropout_p)
        # self.cand_doc_encoder = TransformerModel(nemb=embedding_dim,
        #                                          nhead=2,
        #                                          nhid=200,
        #                                          nlayers=2,
        #                                          dropout=dropout_p)

    def forward(self, doc):
        doc, _ = del_zeros_right(doc)
        _, sent_right_idx = del_zeros_right(doc.sum(-1))
        doc = doc[:, :sent_right_idx, :]

        # embedding layer
        doc_emb = self.embedding_layer(doc)
        doc_mask = compute_mask(doc)

        # doc encoder layer
        tar_doc_rep, _ = self.tar_doc_encoder(doc_emb, doc_mask)
        cand_doc_rep, _ = self.cand_doc_encoder(doc_emb, doc_mask)

        # doc representation
        doc_rep = torch.cat([tar_doc_rep, cand_doc_rep], dim=-1)

        return doc_rep


class DocRepPTEncoder(torch.nn.Module):
    """
    Documents representation model

    Inputs:
        doc_emb: (batch, doc_sent_len, doc_word_len, emb_dim)
        doc_mask: (batch, doc_sent_len, doc_word_len)

    Outputs:
        doc_rep: (batch, hidden_size * 2)
    """

    def __init__(self, embedding_dim, hidden_size, dropout_p, enable_layer_norm):
        super(DocRepPTEncoder, self).__init__()

        self.hidden_size = hidden_size

        self.dropout_layer = torch.nn.Dropout(p=dropout_p)
        self.doc_word_rnn = MyRNNBase(mode='GRU',
                                      input_size=embedding_dim,
                                      hidden_size=self.hidden_size,
                                      bidirectional=True,
                                      dropout_p=dropout_p,
                                      enable_layer_norm=enable_layer_norm,
                                      batch_first=True,
                                      num_layers=1)
        self.doc_word_attention = SelfAttention(hidden_size=self.hidden_size * 2)

        self.doc_sentence_rnn = MyRNNBase(mode='GRU',
                                          input_size=self.hidden_size * 2,
                                          hidden_size=self.hidden_size,
                                          bidirectional=True,
                                          dropout_p=dropout_p,
                                          enable_layer_norm=enable_layer_norm,
                                          batch_first=True,
                                          num_layers=1)
        self.doc_sentence_attention = SelfAttention(hidden_size=self.hidden_size * 2)

    def forward(self, doc_emb, doc_mask):
        visual_parm = {}
        batch, doc_sent_len, doc_word_len, _ = doc_emb.size()

        doc_word_emb = doc_emb.view(batch * doc_sent_len, doc_word_len, -1)
        doc_word_mask = doc_mask.view(batch * doc_sent_len, doc_word_len)

        # (batch * doc_sent_len, doc_word_len, hidden_size * 2)
        doc_word_rep, _ = self.doc_word_rnn(doc_word_emb, doc_word_mask)

        # (batch * doc_sent_len, hidden_size * 2)
        doc_sent_emb, doc_word_att_p = self.doc_word_attention(doc_word_rep, doc_word_mask)
        visual_parm['doc_word_att_p'] = doc_word_att_p

        # (batch, doc_sent_len, hidden_size * 2)
        doc_sent_emb = doc_sent_emb.view(batch, doc_sent_len, -1)
        doc_sent_mask = compute_top_layer_mask(doc_mask)

        # (batch, doc_sent_len, hidden_size * 2)
        doc_sent_rep, _ = self.doc_sentence_rnn(doc_sent_emb, doc_sent_mask)

        # (batch, hidden_size * 2)
        doc_rep, doc_sent_att_p = self.doc_sentence_attention(doc_sent_rep, doc_sent_mask)
        visual_parm['doc_sent_att_p'] = doc_sent_att_p

        return doc_rep, visual_parm
