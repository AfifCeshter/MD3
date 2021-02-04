#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = "Han"
__email__ = "liuhan132@foxmail.com"

"""Pre-training for Documents Representation"""

import argparse
from tqdm import tqdm
import torch
import torch.nn
import torch.multiprocessing
import logging
from game.agent.doc_rep_module import DocRepTrainModule
from game.dataset_reader import DocPTReader
from utils.functions import get_optimizer
from utils.metrics import evaluate_acc
from utils.config import init_logging, init_env
from pytorch_memlab import MemReporter

logger = logging.getLogger(__name__)


def main(config_path, in_infix, out_infix, slot, is_train, is_test, gpuid):
    logger.info('-------------Doc-Rep Pre-training for %s---------------' % slot)
    logger.info('initial environment...')
    game_config, enable_cuda, device, writer = init_env(config_path, in_infix, out_infix,
                                                        writer_suffix='pt_log_path',
                                                        gpuid=gpuid)

    logger.info('reading dataset...')
    dataset = DocPTReader(game_config)

    logger.info('constructing model...')
    doc_rep_module = DocRepTrainModule(game_config).to(device)
    doc_rep_module.load_parameters(enable_cuda, force=False, strict=True)

    # debug: show using memory
    reporter = MemReporter(doc_rep_module)
    reporter.report()

    # loss function
    criterion = torch.nn.NLLLoss()
    optimizer = get_optimizer(game_config['train']['optimizer'],
                              game_config['train']['learning_rate'],
                              doc_rep_module.parameters())

    # training arguments
    batch_size = game_config['train']['batch_size']
    num_workers = game_config['global']['num_data_workers']
    save_steps = game_config['train']['save_steps']
    train_iters = game_config['train']['train_iters']
    test_iters = game_config['train']['test_iters']

    # dataset loader
    batch_train_data = dataset.get_dataset_train_slot(slot, batch_size, num_workers, train_iters)
    batch_test_data = dataset.get_dataset_test_slot(slot, batch_size, num_workers, test_iters)

    if is_train:
        logger.info('start training...')
        clip_grad_max = game_config['train']['clip_grad_norm']

        # train
        doc_rep_module.train()  # set training = True, make sure right dropout
        train_on_model(model=doc_rep_module,
                       criterion=criterion,
                       optimizer=optimizer,
                       dataloader=batch_train_data,
                       clip_grad_max=clip_grad_max,
                       device=device,
                       writer=writer,
                       save_steps=save_steps)

    if is_test:
        logger.info('start testing...')

        with torch.no_grad():
            doc_rep_module.eval()
            test_acc = eval_on_model(model=doc_rep_module,
                                     dataloader=batch_test_data,
                                     device=device)
        logger.info("test_all_acc=%.2f%%" % (test_acc * 100))

    writer.close()
    logger.info('finished.')


def train_on_model(model, criterion, optimizer, dataloader, clip_grad_max, device, writer, save_steps):
    num_iters = len(dataloader)
    for step_i, batch in tqdm(enumerate(dataloader), total=len(dataloader), desc='Training...'):
        step_i += 1
        optimizer.zero_grad()

        # batch data
        batch = [x.to(device) if x is not None else x for x in batch]
        cls_truth = batch[-1]
        batch_input = batch[:-1]

        # forward
        cls_predict = model.forward(*batch_input)

        loss = criterion(cls_predict, cls_truth)
        loss.backward()

        # evaluate
        batch_acc, batch_eq_num = evaluate_acc(cls_predict, cls_truth)

        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_max)  # fix gradient explosion
        optimizer.step()  # update parameters

        # logging
        batch_loss = loss.item()
        writer.add_scalar('Train-Step-Loss', batch_loss, global_step=step_i)
        writer.add_scalar('Train-Step-Acc', batch_acc, global_step=step_i)

        if step_i % save_steps == 0 or step_i == num_iters:
            logger.debug('Steps %d: loss=%.5f, acc=%.2f%%' % (step_i, batch_loss, batch_acc * 100))
            model.save_parameters(step_i)


def eval_on_model(model, dataloader, device):
    eq_num = 0
    all_num = 0

    for batch in tqdm(dataloader, desc='Testing...'):
        # batch data
        batch = [x.to(device) if x is not None else x for x in batch]
        cls_truth = batch[-1]
        batch_input = batch[:-1]

        # forward
        cls_predict = model.forward(*batch_input)
        batch_acc, batch_eq_num = evaluate_acc(cls_predict, cls_truth)

        batch_num = cls_truth.shape[0]
        eq_num += batch_eq_num
        all_num += batch_num

    acc = eq_num / all_num
    return acc


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--in', dest='in_infix', type=str, default='default', help='input path infix')
    parser.add_argument('--out', type=str, default='default', help='output path infix')
    parser.add_argument('--slot', type=str, default='directed_by', help='output path infix')
    parser.add_argument('--train', action='store_true', default=False, help='enable train step')
    parser.add_argument('--test', action='store_true', default=False, help='enable test step')
    parser.add_argument('--gpuid', type=int, default=None, help='gpuid')
    args = parser.parse_args()

    in_infix = args.in_infix + '/' + args.slot
    out_infix = args.out + '/' + args.slot

    init_logging(out_infix=out_infix)
    main('config/game_config.yaml', in_infix=in_infix, out_infix=out_infix, slot=args.slot,
         is_train=args.train, is_test=args.test, gpuid=args.gpuid)
