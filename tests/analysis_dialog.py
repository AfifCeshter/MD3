#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = "Han"
__email__ = "liuhan132@foxmail.com"

import json
import numpy as np
from tqdm import tqdm
from utils.functions import draw_heatmap_sea


def show_doc_dynamic(data_path, max_turns, infix):
    with open(data_path, 'r') as f:
        dialog_data = json.load(f)

    doc_dynamic_rank = np.zeros([len(dialog_data), max_turns-1], dtype=np.float)
    doc_dynamic_entropy = np.zeros([len(dialog_data), max_turns-1], dtype=np.float)

    for dialog_i, ele in tqdm(enumerate(dialog_data), total=len(dialog_data)):
        dialog = ele['dialog']
        for turn_i, turn in enumerate(dialog[:-1]):
            doc_dynamic_rank[dialog_i][turn_i] = turn['tar_rank']
            doc_dynamic_entropy[dialog_i][turn_i] = turn['docs_entropy']

    draw_heatmap_sea(doc_dynamic_rank.T,
                     # title='Target Document Order',
                     title=None,
                     ylabels=[i + 1 for i in range(max_turns - 1)],
                     xlabels='',
                     cmap='YlGnBu',
                     vmax=10,
                     vmin=0,
                     save_path='data/imgs/docs_rank_dynamics_{}.eps'.format(infix),
                     inches=(4, 2), left=0.1, right=0.999)

    draw_heatmap_sea(doc_dynamic_entropy[:, :].T,
                     # title='Document Entropy Dynamics',
                     title=None,
                     ylabels=[i + 1 for i in range(max_turns - 1)],
                     xlabels='',
                     cmap='YlGnBu',
                     # center=2,
                     vmin=0,
                     vmax=3.4,
                     save_path='data/imgs/docs_entropy_dynamics_{}.eps'.format(infix),
                     inches=(4, 2), left=0.1, right=0.999)

    return doc_dynamic_rank,  doc_dynamic_entropy


if __name__ == '__main__':
    mdam_mddp_path = 'outputs/20200225/dialog-model-rl-32-8-test-5k/dialog_data.json'
    mdam_rand_path = 'outputs/20200225/dialog-model-rand-32-8-5k/dialog_data.json'
    mdam_fixed_path = 'outputs/20200225/dialog-model-fixed-1-32-8-5k/dialog_data.json'
    mdam_no_diff_path = 'outputs/20200225/dialog-model-no-diff-32-8-5k/dialog_data.json'
    mdam_no_dst_path = 'outputs/20200225/dialog-model-no-dst-32-8-5k/dialog_data.json'
    mrc_rand_path = 'outputs/20200304/dialog-mrc-rand/dialog_data.json'
    mrc_fixed_path = 'outputs/20200304/dialog-mrc-fixed-1/dialog_data.json'

    show_doc_dynamic(data_path=mdam_mddp_path,
                     max_turns=5,
                     infix='mdam_mddp')
    show_doc_dynamic(data_path=mdam_rand_path,
                     max_turns=5,
                     infix='mdam_rand')
    show_doc_dynamic(data_path=mrc_rand_path,
                     max_turns=5,
                     infix='mrc_rand')
