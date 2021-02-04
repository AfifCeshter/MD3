#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = "Han"
__email__ = "liuhan132@foxmail.com"

"""Simulate the GuessMovie game"""


import argparse
import logging
from game.dialog_manager import MultiDM
from utils.config import init_logging, init_env

logger = logging.getLogger(__name__)


def run(config_path, in_infix, out_infix, is_train, is_test, gpuid):
    logger.info('-------------GuessMovie Simulating---------------')
    logger.info('initial environment...')
    game_config, enable_cuda, device, writer = init_env(config_path, in_infix, out_infix,
                                                        writer_suffix='main_log_path',
                                                        gpuid=gpuid)

    # dialog manager
    dialog_manager = MultiDM(game_config, writer, device)

    if is_train:
        logger.info('Training start...')
        dialog_manager.train_dataset()
        logger.info('Training end.')

    if is_test:
        logger.info('Testing start')
        ave_top1_success, ave_top3_success, ave_mrr, ave_turns, ave_rewards = dialog_manager.test_dataset()

        logger.info('Ave Top-1 success=%.2f' % (ave_top1_success * 100))
        logger.info('Ave Top-3 success=%.2f' % (ave_top3_success * 100))
        logger.info('Ave mrr=%.3f' % ave_mrr)
        logger.info('Ave dialog turns=%.2f' % ave_turns)
        logger.info('Ave rewards=%.2f' % ave_rewards)

        logger.info('Testing end.')

    writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--in', dest='in_infix', type=str, default='default', help='input path infix')
    parser.add_argument('--out', type=str, default='default', help='output path infix')
    parser.add_argument('--train', action='store_true', default=False, help='enable train step')
    parser.add_argument('--test', action='store_true', default=False, help='enable test step')
    parser.add_argument('--gpuid', type=int, default=None, help='gpuid')
    args = parser.parse_args()

    init_logging(out_infix=args.out)
    run(config_path='config/game_config.yaml', in_infix=args.in_infix, out_infix=args.out,
        is_train=args.train, is_test=args.test, gpuid=args.gpuid)
