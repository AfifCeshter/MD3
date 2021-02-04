#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = "Han"
__email__ = "liuhan132@foxmail.com"


import argparse
from tqdm import tqdm
import torch
import torch.nn
import torch.multiprocessing
import logging
from game.dataset_reader import DocPTReader
from utils.functions import set_seed
from utils.config import init_logging, read_config
from game.dataset_reader import load_docs_rep

logger = logging.getLogger(__name__)


def main(config_path, out_infix, slot):
    logger.info('loading config file...')
    game_config = read_config(config_path, out_infix)

    # set multi-processing: bugs in `list(dataloader)`
    # see more on `https://github.com/pytorch/pytorch/issues/973`
    torch.multiprocessing.set_sharing_strategy('file_system')

    # set random seed
    set_seed(game_config['global']['random_seed'])

    logger.info('reading dataset...')
    dataset = DocPTReader(game_config)

    # training arguments
    batch_size = 1
    num_workers = 5
    test_iters = 500

    # dataset loader
    batch_test_data = dataset.get_dataset_test_slot(slot, batch_size, num_workers, test_iters)
    docs_name = dataset.doc_reader.get_all_names()

    logger.info('start testing...')

    with torch.no_grad():
        test_mrr = eval_on_rep(docs_name, batch_test_data)
    logger.info("test_all_mrr=%.2f%%" % test_mrr)
    logger.info('finished.')


def eval_on_rep(docs_name, dataloader):
    docs_rep = load_docs_rep('data/doc_rep/pt/dialog_doc_pt_rep.pt')

    mrr = []

    for batch in tqdm(dataloader, desc='Testing...'):
        _, _, batch_ground_truth_idx, batch_cand_doc_names = batch
        ground_truth_idx = batch_ground_truth_idx[0].item()
        cand_doc_names = batch_cand_doc_names[0]

        cand_idx = [docs_name.index(name) for name in cand_doc_names]
        tar_idx = cand_idx[ground_truth_idx]

        cand_docs_rep = docs_rep[cand_idx, 0, 400:]
        tar_rep = docs_rep[tar_idx, 0, :400]

        cand_prob = torch.mm(tar_rep.unsqueeze(0), cand_docs_rep.transpose(0, 1)).squeeze(0)
        cand_prob_softmax = torch.softmax(cand_prob, dim=-1)

        sort_prob, sort_idx = torch.sort(cand_prob_softmax, dim=-1, descending=True)
        cur_mrr = 1.0 / (sort_idx.tolist().index(ground_truth_idx) + 1)
        mrr.append(cur_mrr)

    return sum(mrr) * 1.0 / len(mrr)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=str, default='default', help='output path infix')
    parser.add_argument('--slot', type=str, default='directed_by', help='output path infix')
    args = parser.parse_args()

    out_infix = args.out

    init_logging(out_infix=out_infix)
    main('config/game_config.yaml', out_infix=out_infix, slot=args.slot)