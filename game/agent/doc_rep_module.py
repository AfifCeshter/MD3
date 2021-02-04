#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = "Han"
__email__ = "liuhan132@foxmail.com"

import os
import logging
import torch
from game.models import DocRepPTTrainModel, DocRepPTTestModel
from utils.functions import save_model, load_checkpoint_parameters

logger = logging.getLogger(__name__)


class DocRepModule(torch.nn.Module):
    def __init__(self, game_config):
        super(DocRepModule, self).__init__()
        self.game_config = game_config

        self.in_checkpoint_path = game_config['checkpoint']['in_pt_checkpoint_path']
        self.in_model_weight_path = game_config['checkpoint']['in_pt_weight_path']
        self.out_pt_checkpoint_path = game_config['checkpoint']['out_pt_checkpoint_path']
        self.out_pt_weight_path = game_config['checkpoint']['out_pt_weight_path']

        self.doc_rep_model = self.get_model()

    def get_model(self):
        return NotImplementedError

    def forward(self, *args):
        return self.doc_rep_model(*args)

    def load_parameters(self, enable_cuda, force=False, strict=False):
        if force:
            assert os.path.exists(self.in_checkpoint_path)

        if os.path.exists(self.in_checkpoint_path):
            logger.info('loading parameters for doc-rep module')
            load_weight_path = load_checkpoint_parameters(self,
                                                          self.in_model_weight_path,
                                                          self.in_checkpoint_path,
                                                          enable_cuda,
                                                          strict)
            logger.info('loaded doc-rep module from %s' % load_weight_path)

    def save_parameters(self, num):
        """
        Save the trained parameters
        :param num:
        :return:
        """
        logger.info('saving parameters for doc-rep module on step=%d' % num)
        save_model(self,
                   num,
                   model_weight_path=self.out_pt_weight_path + '-' + str(num),
                   checkpoint_path=self.out_pt_checkpoint_path)


class DocRepTrainModule(DocRepModule):
    def __init__(self, game_config):
        super(DocRepTrainModule, self).__init__(game_config)

    def get_model(self):
        return DocRepPTTrainModel(self.game_config['model'],
                                  embedding_path=self.game_config['dataset']['embedding_path'],
                                  embedding_freeze=self.game_config['dataset']['embedding_freeze'])


class DocRepTestModule(DocRepModule):
    def __init__(self, game_config):
        super(DocRepTestModule, self).__init__(game_config)

    def get_model(self):
        return DocRepPTTestModel(self.game_config['model'],
                                 embedding_path=self.game_config['dataset']['embedding_path'],
                                 embedding_freeze=self.game_config['dataset']['embedding_freeze'])
