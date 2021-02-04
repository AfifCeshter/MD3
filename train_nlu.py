#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = "Han"
__email__ = "liuhan132@foxmail.com"

"""Train the NLU module"""

import argparse
import torch
import torch.nn
import logging
from tqdm import tqdm
from game.models import StateNLLLoss
from game.agent.state_module import NeuralState
from game.dataset_reader import ImitationStateReader
from utils.functions import get_optimizer
from utils.metrics import evaluate_top_k, evaluate_MAP, evaluate_acc, evaluate_acc_sigmoid
from utils.config import init_logging, init_env

logger = logging.getLogger(__name__)


def main(config_path, in_infix, out_infix, is_train, is_test, gpuid):
    logger.info('-------------GuessMovie Supervised Training for NLU---------------')
    logger.info('initial environment...')
    game_config, enable_cuda, device, writer = init_env(config_path, in_infix, out_infix,
                                                        writer_suffix='state_log_path',
                                                        gpuid=gpuid)

    logger.info('reading dataset...')
    dataset = ImitationStateReader(game_config)

    logger.info('constructing model...')
    model = NeuralState(game_config).to(device)
    model.load_parameters(enable_cuda)

    # loss function
    criterion = StateNLLLoss()
    slot_cls_criterion = torch.nn.NLLLoss()
    inform_cls_criterion = torch.nn.BCELoss()

    optimizer = get_optimizer(game_config['train']['optimizer'],
                              game_config['train']['learning_rate'],
                              model.parameters())

    # training arguments
    batch_size = game_config['train']['batch_size']
    num_workers = game_config['global']['num_data_workers']
    save_steps = game_config['train']['save_steps']
    train_iters = game_config['train']['train_iters']
    test_iters = game_config['train']['test_iters']

    # dataset loader
    batch_train_data = dataset.get_dataloader_train(batch_size, num_workers, train_iters)
    batch_test_data = dataset.get_dataloader_test(batch_size, num_workers, test_iters)

    if is_train:
        logger.info('start training...')

        clip_grad_max = game_config['train']['clip_grad_norm']

        # train
        model.train()  # set training = True, make sure right dropout
        train_on_model(model=model,
                       criterion=criterion,
                       slot_cls_criterion=slot_cls_criterion,
                       inform_cls_criterion=inform_cls_criterion,
                       optimizer=optimizer,
                       dataloader=batch_train_data,
                       clip_grad_max=clip_grad_max,
                       device=device,
                       writer=writer,
                       save_steps=save_steps)

    if is_test:
        logger.info('start testing...')

        with torch.no_grad():
            model.eval()
            metrics = eval_on_model(model=model,
                                    dataloader=batch_test_data,
                                    device=device)
        logger.info("P_1=%.3f, P_3=%.3f, MAP=%.3f, S_Acc=%.3f, I_Acc=%.3f"
                    % (metrics['ave_top_p1'], metrics['ave_top_p2'],
                       metrics['ave_map'], metrics['user_slot_acc'], metrics['user_inform_acc']))

    writer.close()
    logger.info('finished.')


def train_on_model(model, criterion, slot_cls_criterion, inform_cls_criterion, optimizer, dataloader, clip_grad_max,
                   device, writer, save_steps):
    num_iters = len(dataloader)
    for step_i, batch in tqdm(enumerate(dataloader), total=len(dataloader), desc='Training...'):
        step_i += 1
        optimizer.zero_grad()

        # batch data
        batch = [x.to(device) if x is not None else x for x in batch]
        is_truth, slot_gt, inform_gt = batch[-3:]
        batch_input = batch[:-3]

        # forward
        predict_doc_dist, turn_slot_cls, turn_inform_cls = model.get_internal_state(*batch_input)
        turn_slot_log = torch.log(turn_slot_cls)
        turn_inform_cls = turn_inform_cls.squeeze(-1)

        state_loss = criterion(predict_doc_dist, is_truth)
        slot_loss = slot_cls_criterion(turn_slot_log, slot_gt)
        inform_loss = inform_cls_criterion(turn_inform_cls, inform_gt)

        loss = state_loss + slot_loss + inform_loss
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_max)  # fix gradient explosion
        optimizer.step()  # update parameters

        # evaluate
        _, predict_sort_idx = torch.sort(predict_doc_dist, dim=-1, descending=True)
        top_p1 = evaluate_top_k(predict_sort_idx, is_truth, k=1, reduce=True)
        top_p2 = evaluate_top_k(predict_sort_idx, is_truth, k=3, reduce=True)
        metric_map = evaluate_MAP(predict_sort_idx, is_truth, reduce=True)

        user_slot_acc, _ = evaluate_acc(turn_slot_cls, slot_gt)
        user_inform_acc, _ = evaluate_acc_sigmoid(turn_inform_cls, inform_gt)
        has_inform_prop = inform_gt.mean().item()

        # logging
        batch_loss = loss.item()
        writer.add_scalar('Train-Step-Loss', batch_loss, global_step=step_i)
        writer.add_scalar('Train-Step-P_1', top_p1, global_step=step_i)
        writer.add_scalar('Train-Step-P_3', top_p2, global_step=step_i)
        writer.add_scalar('Train-Step-MAP', metric_map, global_step=step_i)
        writer.add_scalar('Train-Step-S_Acc', user_slot_acc, global_step=step_i)
        writer.add_scalar('Train-Step-I_Acc', user_inform_acc, global_step=step_i)
        writer.add_scalar('Train-Step-I_Prop', has_inform_prop, global_step=step_i)

        if step_i % save_steps == 0 or step_i == num_iters:
            logger.debug('Steps %d: loss=%.5f, P_1=%.3f, P_3=%.3f, MAP=%.3f, S_Acc=%.3f, I_Acc=%.3f' %
                         (step_i, batch_loss, top_p1, top_p2, metric_map, user_slot_acc, user_inform_acc))
            model.save_parameters(step_i)


def eval_on_model(model, dataloader, device):
    all_top_p1 = []
    all_top_p2 = []
    all_map = []
    all_user_slot_eq_num = 0
    all_user_inform_eq_num = 0
    all_num = 0

    for batch in tqdm(dataloader, desc='Testing...'):
        # batch data
        batch = [x.to(device) if x is not None else x for x in batch]
        is_truth, slot_gt, inform_gt = batch[-3:]
        batch_input = batch[:-3]

        # forward
        predict_doc_dist, turn_slot_cls, turn_inform_cls = model.get_internal_state(*batch_input)
        turn_inform_cls = turn_inform_cls.squeeze(-1)

        # evaluate
        _, predict_sort_idx = torch.sort(predict_doc_dist, dim=-1, descending=True)
        # _, predict_sort_idx = torch.sort(is_truth, dim=-1, descending=True)
        top_p1 = evaluate_top_k(predict_sort_idx, is_truth, k=1, reduce=False)
        all_top_p1.append(top_p1)

        top_p2 = evaluate_top_k(predict_sort_idx, is_truth, k=3, reduce=False)
        all_top_p2.append(top_p2)

        metric_map = evaluate_MAP(predict_sort_idx, is_truth, reduce=False)
        all_map.append(metric_map)

        _, user_slot_eq_num = evaluate_acc(turn_slot_cls, slot_gt)
        all_user_slot_eq_num += user_slot_eq_num

        _, user_inform_eq_num = evaluate_acc_sigmoid(turn_inform_cls, inform_gt)
        all_user_inform_eq_num += user_inform_eq_num

        batch_num = is_truth.shape[0]
        all_num += batch_num

    all_top_p1 = torch.cat(all_top_p1, dim=0)
    all_top_p2 = torch.cat(all_top_p2, dim=0)
    all_map = torch.cat(all_map, dim=0)
    user_slot_acc = all_user_slot_eq_num * 1.0 / all_num
    user_inform_acc = all_user_inform_eq_num * 1.0 / all_num

    metrics = {'ave_top_p1': all_top_p1.mean().item(),
               'ave_top_p2': all_top_p2.mean().item(),
               'ave_map': all_map.mean().item(),
               'user_slot_acc': user_slot_acc,
               'user_inform_acc': user_inform_acc}

    return metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--in', dest='in_infix', type=str, default='default', help='input path infix')
    parser.add_argument('--out', type=str, default='default', help='output path infix')
    parser.add_argument('--train', action='store_true', default=False, help='enable train step')
    parser.add_argument('--test', action='store_true', default=False, help='enable test step')
    parser.add_argument('--gpuid', type=int, default=None, help='gpuid')
    args = parser.parse_args()

    init_logging(out_infix=args.out)
    main('config/game_config.yaml', args.in_infix, args.out,
         is_train=args.train, is_test=args.test, gpuid=args.gpuid)
