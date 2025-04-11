import json
import itertools
import random
import pandas as pd
import numpy as np

import sampling as spl

from tqdm import tqdm

# TODO : optimize load_iam fn -> if truly necessary

'''
# def get_lbls(lst_s: list[dict]) -> set:
#     lbls = [
#         i
#         for s in lst_s
#         for i in s.get('label')
#     ]
#     return set(lbls)

# def get_all_labels(data: dict) -> set:
#     labels = set()
#     for k,v in data.items():
#         labels = labels.union(v.get('list_label'))
#     return labels

# def get_nb_element_by_class(lbls: set, data: list[dict]) -> dict:
#     res = {}
#     for l in lbls:
#         tmp = [
#             x
#             for x in data
#             if l in x.get('label')
#         ]
#         res.update({l: len(tmp)})
#     return res

# def get_train_val_test_split(
#     data: list[dict],
#     lbls: set,
#     val_size: float=0.2,
#     test_size: float=0.2
# ) -> tuple[list[dict], list[dict], list[dict]]:
#     test_sample = []
#     validation_sample = []
#     n_val = len(data)*val_size
#     n_test = len(data)*test_size
#     nb_elmt = get_nb_element_by_class(lbls, data)
#     data = pd.Series(data)
#     for l in lbls:
#         ratio = nb_elmt.get(l) / sum(nb_elmt.values())
#         n_sample_val = int(ratio*n_val)
#         n_sample_test = int(ratio*n_test)
#         tmp = [x for x in data if l in x.get('label')]
#         sample_val = random.sample(tmp, n_sample_val)
#         tmp = [x for x in tmp if x not in sample_val]
#         sample_test = random.sample(tmp, n_sample_test)
#         validation_sample.extend(sample_val)
#         test_sample.extend(sample_test)
#     validation_sample = random.sample(validation_sample, len(validation_sample))
#     test_sample = random.sample(test_sample, len(test_sample))
#     train_sample = [
#         s
#         for s in data
#         if s not in test_sample and s not in validation_sample
#     ]
#     return train_sample, validation_sample, test_sample
'''

def load_iam_claim(path:str='./Data_jsonl/iam_claim.jsonl') -> dict:
    all_data = []
    sentences = []
    with open(path, 'r') as f:
        for line in f:
            all_data.append(json.loads(line))
    for data in all_data:
        tmp = {
            'topic': data.get('topic'),
            'sentences': data.get('sentence'),
            'label': data.get('label_claim').split(','),
        }
        sentences.append(tmp)
    lbl_iam_claim = spl.get_lbls(sentences)
    train_set, validation_set, test_set = spl.get_train_val_test_split(
        data=sentences,
        lbls=lbl_iam_claim
    )
    res = {
        'train': train_set,
        'validation': validation_set,
        'test': test_set,
        'list_label': lbl_iam_claim
    }
    return res

def load_ibm_claim(path:str='./Data_jsonl/ibm_claim.jsonl') -> dict:
    all_data = []
    sentences = []
    with open(path, 'r') as f:
        for line in f:
            all_data.append(json.loads(line))
    for data in all_data:
        tmp = {
            'topic': data.get('topic'),
            'sentences': data.get('sentence'),
            'label': data.get('label').split(','),
        }
        sentences.append(tmp)
    lbl_ibm_claim = spl.get_lbls(sentences)
    train_set, validation_set, test_set = spl.get_train_val_test_split(
        data=sentences,
        lbls=lbl_ibm_claim
    )
    res = {
        'train': train_set,
        'validation': validation_set,
        'test': test_set,
        'list_label': lbl_ibm_claim
    }
    return res

def load_ibm_argument(path:str='./Data_jsonl/ibm_argument.jsonl') -> dict:
    all_data = []
    sentences = []
    with open(path, 'r') as f:
        for line in f:
            all_data.append(json.loads(line))
    for data in all_data:
        tmp = {
            'topic': data.get('topic'),
            'sentence': data.get('sentence'),
            'text': data.get('context'),
            'label': data.get('label').split(','),
        }
        sentences.append(tmp)
    lbl_ibm_args = spl.get_lbls(sentences)
    train_set, validation_set, test_set = spl.get_train_val_test_split(
        data=sentences,
        lbls=lbl_ibm_args
    )
    res = {
        'train': train_set,
        'validation': validation_set,
        'test': test_set,
        'list_label': lbl_ibm_args
    }
    return res

def load_all_datasets(paths: dict) -> tuple[dict, set]:
    res = {
        'iam_claim': load_iam_claim(paths.get('iam_claim')),
        'ibm_claim': load_ibm_claim(paths.get('ibm_claim')),
        'ibm_args': load_ibm_argument(paths.get('ibm_args')),
    }
    labels = spl.get_all_labels(res)
    return res, labels