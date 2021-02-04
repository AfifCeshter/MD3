#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = "Han"
__email__ = "liuhan132@foxmail.com"

import itertools
import json
import random
from functools import reduce

import matplotlib.pyplot as plt
import nltk
from tqdm import tqdm

random.seed(1)


def analysis_data(doc_path, attr_path=None):
    with open(doc_path, 'r') as f:
        movie_doc_kb = json.load(f)

    print(len(movie_doc_kb))

    if attr_path is not None:
        attr_keys = {}
        for ele in movie_doc_kb:
            for k in ele['kb'].keys():
                if k not in attr_keys:
                    attr_keys[k] = 0

                attr_keys[k] += 1

        with open(attr_path, 'w') as af:
            tmp_keys = list(map(
                lambda x: str(x) + '\n',
                sorted(attr_keys.items(), key=lambda y: y[1], reverse=True)
            )
            )
            af.writelines(tmp_keys)

        print('Attributes Keys: ', tmp_keys)


def analysis_doc_len(doc_path):
    with open(doc_path, 'r') as f:
        movie_doc_kb = json.load(f)

    doc_lens = []
    for ele in movie_doc_kb:
        doc_lens.append(len(ele['doc']))

    plt.scatter(range(len(doc_lens)), doc_lens)
    print(min(doc_lens), sum(doc_lens) / len(doc_lens), max(doc_lens))


def analysis_qa_len():
    with open('data/guessmovie_dialog.json', 'r') as f:
        dialog_data = json.load(f)

    qa_lens = []
    for ele in dialog_data:
        nl = ''
        for turn in ele['dialog']:
            nl += turn['agent_nl']
            nl += ' . '
            nl += turn['user_nl']

        qa_lens.append(len(nltk.word_tokenize(nl)))

    plt.scatter(range(len(qa_lens)), qa_lens)
    print(min(qa_lens), sum(qa_lens) / len(qa_lens), max(qa_lens))


# def analysis_turn_docs_data():
#     game_config = read_config('config/game_config.yaml')
#     dataset = DialogDocReader(game_config)
#     batch_train_data = dataset.get_dataloader_train(128, 10)
#     batch_dev_data = dataset.get_dataloader_dev(128, 10)
#     batch_test_data = dataset.get_dataloader_test(128, 10)
#
#     is_valid = []
#     for batch in tqdm(batch_test_data):
#         valid_truth = batch[-1]
#         valid_truth = valid_truth.tolist()
#         is_valid += valid_truth
#
#     valid_per = sum(is_valid) / len(is_valid)
#     print('valid percentage: %.2f%%' % (valid_per * 100))


def count_same_kb(in_path='data/guessmovie_same_name.json'):
    with open(in_path, 'r') as f:
        data = json.load(f)

    cnums = []
    snums = []

    for mv in tqdm(data):
        kb_num = len(mv['same_kb_name'])

        for combine_num in range(kb_num, 0, -1):
            for combine_kbs in itertools.combinations(mv['same_kb_name'].items(), combine_num):
                and_keys = []
                and_names = []

                for kb in combine_kbs:
                    and_keys.append(kb[0])
                    and_names.append(kb[1])

                and_names = list(reduce(lambda x, y: set(x) & set(y), and_names))

                cnums.append(combine_num)
                snums.append(len(and_names))

    for combine_num in range(7, 0, -1):
        filter_snum = filter(lambda x: x[0] == combine_num,
                             zip(cnums, snums))
        filter_snum = map(lambda x: x[1], filter_snum)
        max_snum = max(filter_snum)

        print(combine_num, max_snum)

    plt.scatter(cnums, snums)


def count_same_kb_per_movie(in_path='data/guessmovie_same_name.json'):
    with open(in_path, 'r') as f:
        data = json.load(f)

    all_nums = []
    for ele in tqdm(data):
        same_kb_name = ele['same_kb_name']
        same_kb_name_nums = reduce(
            lambda x, y: x | y,
            map(lambda m: set(m[1]), same_kb_name.items())
        )

        all_nums.append(len(same_kb_name_nums))
    print(min(all_nums), sum(all_nums) / len(all_nums), max(all_nums))


def analysis_len():
    with open('/Users/bytedance/GuessMovie/data/guessmovie_doc.json', 'r') as f:
        data = json.load(f)

    all_len = 0
    for sample in tqdm(data):
        cur_len = len(nltk.word_tokenize(sample['doc']))
        all_len += cur_len

    print(all_len / len(data))


def analysis_turns():
    with open('/Users/bytedance/GuessMovie/data/guessmovie_dialog.json', 'r') as f:
        data = json.load(f)

    all_turns = 0
    for sample in tqdm(data):
        all_turns += len(sample['dialog'])

    print(all_turns / len(data))


def analysis_multi_value():
    with open('/Users/bytedance/GuessMovie/data/guessmovie_doc.json', 'r') as f:
        data = json.load(f)

    multi_value = {}
    num_value = {}
    k_count = {}

    for sample in data:
        for k, v in sample['kb'].items():
            if len(v) > 1:
                multi_value[k] = multi_value.get(k, 0) + 1
            num_value[k] = num_value.get(k, 0) + len(v)
            k_count[k] = k_count.get(k, 0) + 1

    for k in multi_value.keys():
        print(k, multi_value[k] / k_count[k])
        print(k, num_value[k] / k_count[k])

    # print(multi_value, num_value, k_count)


def sample_json():
    with open('/Users/bytedance/GuessMovie/data/guessmovie_doc.json', 'r') as f:
        data = json.load(f)

    samples = random.sample(data, 50)
    with open('/Users/bytedance/GuessMovie/data/guessmovie_samples.json', 'w') as wf:
        json.dump(samples, wf, indent=2, ensure_ascii=False)


def make_dialog():
    with open('/Users/bytedance/GuessMovie/outputs/human-cand/dialog_data.json', 'r') as f:
        data = json.load(f)

    print(len(data))
    # samples = list(filter(lambda x: x['top_1_success'], data))
    samples = list(filter(lambda x: x['dialog'][-1]['agent_value'] == x['tar_name'] and \
                                    x['top_1_success'], data))
    print(len(samples))

    # samples = random.sample(samples, 5000)
    for ele in samples:
        for turn in ele['dialog']:
            del turn['docs_entropy']
            del turn['docs_dist']

    with open('/Users/bytedance/GuessMovie/data/guessmovie_dialog.json', 'w') as wf:
        json.dump(samples, wf, indent=2, ensure_ascii=False)


def sample_dialog():
    with open('/Users/bytedance/GuessMovie/data/guessmovie_dialog.json', 'r') as f:
        data = json.load(f)

    samples = random.sample(data, 50)
    with open('/Users/bytedance/GuessMovie/data/guessmovie_dialog_samples.json', 'w') as wf:
        json.dump(samples, wf, indent=2, ensure_ascii=False)
        

def get_entropy():
    in_path = '/Users/bytedance/GuessMovie/data/guessmovie_doc.json'

    entity_num = {}
    with open(in_path, 'r') as f:
        data = json.load(f)
        print(len(data))

    for item in data:
        for att, entity in item['kb'].items():
            for e in entity:
                entity_num[att] = entity_num.get(att, set())
                entity_num[att].add(e)

    for att, entity_set in entity_num.items():
        print(att, len(entity_set), len(entity_set) / len(data))


if __name__ == '__main__':
    analysis_data(doc_path='data/guessmovie_doc.json',
                  attr_path='data/attributes_keys.txt')
    # analysis_doc_len(doc_path='data/doc_vocab_no_entity/guessmovie_doc_id.json')
    # analysis_qa_len()
    # plt.show()

    # analysis_turn_docs_data()
    # count_same_kb_per_movie()

    # analysis_data(doc_path='data/doc_rep/guessmovie_dialog_doc_id.json')
