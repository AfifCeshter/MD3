#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = "Han"
__email__ = "liuhan132@foxmail.com"


"""Preprocess the GuessMovie dataset"""

import os
import argparse
from game.tokenizer import PreProcess
from utils.config import read_config
from utils.functions import set_seed, split_dataset, detect_same_kb, detect_unk_kb


def split_two(doc_id_path, out_path_prefix, per=0.7):
    if not os.path.exists(out_path_prefix + 'doc_rep'):
        os.makedirs(out_path_prefix + 'doc_rep')

    rep_doc_path = out_path_prefix + 'doc_rep/guessmovie_rep_doc_id.json'
    rep_same_path = out_path_prefix + 'doc_rep/rep_doc_same_name.json'
    rep_unk_path = out_path_prefix + 'doc_rep/rep_doc_unk_name.json'

    dialog_doc_path = out_path_prefix + 'guessmovie_dialog_doc_id.json'
    dialog_same_path = out_path_prefix + 'dialog_doc_same_name.json'
    dialog_unk_path = out_path_prefix + 'dialog_doc_unk_name.json'

    print('spliting documents...')
    rep_doc_num, _, dialog_doc_num = split_dataset(doc_id_path, rep_doc_path,
                                                   dev_path=None, test_path=dialog_doc_path,
                                                   dev_per=0, test_per=1 - per)
    print('#Rep documents num:', rep_doc_num)
    print('#Dialog documents num:', dialog_doc_num)

    detect_same_kb(data_path=rep_doc_path, out_path=rep_same_path)
    detect_unk_kb(data_path=rep_doc_path, out_path=rep_unk_path)
    detect_same_kb(data_path=dialog_doc_path, out_path=dialog_same_path)
    detect_unk_kb(data_path=dialog_doc_path, out_path=dialog_unk_path)


def run(config_path, is_vocab, is_split):
    game_config = read_config(config_path)
    set_seed(game_config['global']['random_seed'])

    if is_vocab:
        print('Pre-Processing GuessMovie dataset...')
        preprocessor = PreProcess(out_path='data/wo_entity/vocab/')
        preprocessor.pre_process_entity(replace_ent=False)

    if is_split:
        print('Spliting dataset...')
        split_two(doc_id_path='data/wo_entity/vocab/guessmovie_doc_id.json',
                  out_path_prefix='data/wo_entity/')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--vocab', action='store_true',
                        default=False, help='enable preprocess step')
    parser.add_argument('--split', action='store_true',
                        default=False, help='split the dataset')
    args = parser.parse_args()

    # only one action is allowed
    assert int(args.split) + int(args.vocab) == 1

    run('config/game_config.yaml', args.vocab, args.split)
