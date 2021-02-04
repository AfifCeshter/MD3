#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = "Han"
__email__ = "liuhan132@foxmail.com"

import os
import sys
import argparse
import seaborn as sns
import matplotlib.pyplot as plt
import re

sys.path.append(os.getcwd())

TRAIN_SIZE = 11137
DEV_SIZE = 1391

sns.set_style("dark")


def analysis_log_loss(log_txt, type='train'):
    epoch = []
    loss = []
    if type == 'train':
        p = re.compile(r'.*epoch=(\d*), sum_loss=(\d*\.\d*).*')
    elif type == 'dev':
        p = re.compile(r'.*epoch=(\d*), valid_sum_loss=(\d*\.\d*).*')
    else:
        raise ValueError(type)

    for line in log_txt:
        result = re.findall(p, line)
        if len(result) == 0:
            continue

        epoch.append(int(result[0][0]))

        if type == 'train':
            closs = float(result[0][1]) * 1000 / TRAIN_SIZE
        elif type == 'dev':
            closs = float(result[0][1]) * 1000 / DEV_SIZE
        else:
            raise ValueError(type)

        loss.append(closs)

    return epoch, loss


def draw_loss(epoch, train_loss, eval_loss):
    for i, tl, el in zip(epoch, train_loss, eval_loss):
        print('epoch=%d, train_loss=%f, eval_loss=%f' % (i, tl, el))

    # plot
    x = epoch
    y1 = train_loss
    y2 = eval_loss

    plt.figure()
    plt.plot(x, y1, marker='o', color='b')
    plt.plot(x, y2, marker='^', color='r')

    plt.xlabel('epoch')
    plt.ylabel('loss*1000')
    plt.legend(labels=['train', 'eval'])
    plt.grid()


def main(log_path, x_gen=False):
    with open(log_path) as f_log:
        log_lines = f_log.readlines()
        value_log = log_lines

    epoch, train_loss = analysis_log_loss(value_log, type='train')
    epoch2, dev_loss = analysis_log_loss(value_log, type='dev')

    assert epoch == epoch2, str(epoch) + ' ' + str(epoch2)

    if x_gen:
        epoch = [x for x in range(len(train_loss))]
    draw_loss(epoch, train_loss, dev_loss)

    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="analysis log file that model output")
    parser.add_argument('--log', '-l', required=True, nargs=1, dest='log_path')
    args = parser.parse_args()

    print("analysising log '%s'" % args.log_path[0])
    main(args.log_path[0], x_gen=False)
