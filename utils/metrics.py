#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = "Han"
__email__ = "liuhan132@foxmail.com"

import torch
import torch.nn


def evaluate_acc(predict, truth):
    """
    compute the accuracy of predict value
    :param predict: (batch, _)
    :param truth: (batch)
    :return:
    """
    _, predict_max = predict.max(dim=1)

    batch_eq_num = torch.eq(predict_max, truth).long().sum().item()
    batch_acc = batch_eq_num / truth.shape[0]

    return batch_acc, batch_eq_num


def evaluate_acc_sigmoid(predict, truth):
    """
    accuracy evaluate with two classification on sigmoid
    :param predict: (batch,)
    :param truth: (batch,)
    :return:
    """
    predict_max = predict.gt(0.5).long()

    batch_eq_num = torch.eq(predict_max, truth).long().sum().item()
    batch_acc = batch_eq_num / truth.shape[0]

    return batch_acc, batch_eq_num


def evaluate_acc_sigmoid_turns(predict, truth):
    """
    accuracy evaluate on turns
    :param predict: (batch, num_cand), FloatTensor
    :param truth: (batch, num_cand), LongTensor
    :return:
    """
    predict_max = predict.gt(0.5).long()

    num_cand = predict_max.shape[1]
    is_eq = torch.eq(predict_max, truth)
    batch_eq_num = is_eq.sum(dim=1).eq(num_cand).long().sum().item()

    batch_acc = batch_eq_num / truth.shape[0]
    return batch_acc, batch_eq_num


def evaluate_top_k(predict_sort_idx, is_truth, k, reduce=True):
    """
    Metrics for P@K
    :param predict_sort_idx:
    :param is_truth:
    :param k:
    :param reduce:
    :return:
    """
    top_k_idx = predict_sort_idx[:, :k]
    top_k_is_truth = is_truth.gather(1, top_k_idx)

    batch_acc = top_k_is_truth.sum(dim=-1).float() / k

    if reduce:
        return batch_acc.mean().item()

    return batch_acc


def evaluate_MAP(predict_sort_idx, is_truth, reduce=True):
    """
    Metrics for P@K
    :param predict_sort_idx:
    :param is_truth:
    :param reduce:
    :return:
    """
    # 1. is truth
    predict_sort_is_truth = is_truth.gather(1, predict_sort_idx)

    # 2. rank number
    batch, num_docs = is_truth.size()
    rank_num = torch.arange(1, num_docs+1, device=is_truth.device).unsqueeze(0)

    # 3. top-k truth number
    top_k_truth_num = []
    sum_col = is_truth.new_zeros(batch,)
    for k in range(num_docs):
        col_k = predict_sort_is_truth[:, k]

        sum_col += col_k
        top_k_truth_num.append(sum_col.clone().detach())

    predict_top_k_num = torch.stack(top_k_truth_num, dim=-1)

    batch_map = predict_sort_is_truth.float() * predict_top_k_num.float() / rank_num.float()
    batch_map = batch_map.sum(dim=1) / predict_sort_is_truth.sum(dim=1)

    if reduce:
        return batch_map.mean().item()

    return batch_map


def evaluate_MRR(predict_sort_idx, is_truth, reduce=True):
    return torch.zeros(is_truth.shape[0])
    predict_sort_is_truth = is_truth.gather(1, predict_sort_idx)

    # TODO: sort is not stable
    _, predict_sort_truth_idx = torch.sort(predict_sort_is_truth, descending=True)
    predict_first_truth_idx = predict_sort_truth_idx[:, 0] + 1

    batch_mrr = torch.ones_like(predict_first_truth_idx, dtype=torch.float) / predict_first_truth_idx.float()

    if reduce:
        return batch_mrr.mean().item()

    return batch_mrr
