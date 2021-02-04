#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = "Han"
__email__ = "liuhan132@foxmail.com"

"""pre-build documents representation"""

import argparse
from tqdm import tqdm
import torch
import torch.nn
import torch.multiprocessing
import logging
from game.agent.doc_rep_module import DocRepTestModule
from game.dataset_reader import DocRepPTReader
from utils.config import init_logging, init_env
from pytorch_memlab import MemReporter

logger = logging.getLogger(__name__)


def main(config_path, in_infix, out_infix, slot, gpuid):
    logger.info('-------------Doc-Rep Pre-building for %s---------------' % slot)
    logger.info('initial environment...')
    game_config, enable_cuda, device, writer = init_env(config_path, in_infix, out_infix,
                                                        writer_suffix='pt_log_path',
                                                        gpuid=gpuid)

    logger.info('reading dataset...')
    dataset = DocRepPTReader(game_config)

    logger.info('constructing model...')
    doc_rep_module = DocRepTestModule(game_config).to(device)
    doc_rep_module.load_parameters(enable_cuda, force=True, strict=True)

    # debug: show using memory
    reporter = MemReporter(doc_rep_module)
    reporter.report()

    # training arguments
    batch_size = game_config['train']['batch_size']
    num_workers = game_config['global']['num_data_workers']

    dataloader = dataset.get_dataloader_docs(batch_size, num_workers)

    with torch.no_grad():
        logger.info('start documents encoding...')
        doc_rep_module.eval()
        all_doc_rep = test_on_model(doc_rep_module, dataloader, device)

        logger.info('saving documents vectors...')
        # suffix = '-' + data_type if data_type is not None else ''
        torch.save(all_doc_rep,
                   game_config['dataset']['data_prefix'] + 'doc_rep/pt/dialog_doc_pt_rep.pt-' + slot)

    logger.info('finished.')


def test_on_model(model, dataloader, device):
    all_doc_rep = []

    for batch_doc in tqdm(dataloader, desc='Testing...'):
        batch_doc = batch_doc.to(device)

        # forward
        batch_doc_rep = model.forward(batch_doc)
        all_doc_rep.append(batch_doc_rep)

    all_doc_rep = torch.cat(all_doc_rep, dim=0)
    return all_doc_rep


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--in', dest='in_infix', type=str, default='default', help='input path infix')
    parser.add_argument('--out', type=str, default='default', help='output path infix')
    parser.add_argument('--slot', type=str, default='directed_by', help='output path infix')
    parser.add_argument('--gpuid', type=int, default=None, help='gpuid')
    args = parser.parse_args()

    in_infix = args.in_infix + '/' + args.slot
    out_infix = args.out + '/' + args.slot

    init_logging(out_infix=args.out)
    main('config/game_config.yaml', in_infix=in_infix, out_infix=out_infix, slot=args.slot, gpuid=args.gpuid)
