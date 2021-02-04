#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = "Han"
__email__ = "liuhan132@foxmail.com"

import json


def compare_dialog(data1_path, data2_path):
    with open(data1_path, 'r') as f:
        dialog1_data = json.load(f)

    with open(data2_path, 'r') as f:
        dialog2_data = json.load(f)

    d1_d2_diff = []
    d2_d1_diff = []

    for i in range(len(dialog1_data)):
        d1 = dialog1_data[i]
        d2 = dialog2_data[i]

        if d1['is_success'] == True and d2['is_success'] == False:
            d1_d2_diff.append(d1['tar_name'])

        if d1['is_success'] == False and d2['is_success'] == True:
            d2_d1_diff.append(d1['tar_name'])

    with open('../outputs/d1_d2_diff.json', 'w') as wf:
        json.dump(d1_d2_diff, wf, indent=2)

    with open('../outputs/d2_d1_diff.json', 'w') as wf:
        json.dump(d2_d1_diff, wf, indent=2)