import json
import itertools
import random
import pandas as pd
import numpy as np

import sampling as spl

# The label 'Other' indicates that the edu/adu is neither a Claim or a Premises
# TODO : Add the function for prompting when the file refactoring is done


def get_mt(edu_lst: list[dict]) -> str:
    text = ''
    for edu in edu_lst:
        for v in edu.values():
            text += v
    return text
    
def map_edu_adu(edu: dict, adu: list[dict]) -> dict:
    tmp = []
    res = []
    for k,v in edu.items():
        for a in adu:
            if a.get(k) != None:
                tmp.append(a)
        if len(tmp) == 1:
            for i in tmp:
                elmt = {
                    'sentences': v,
                    'label': i.get('label')
                }
            res.append(elmt)
        else:
            print(f'Multiple adu for one edu\n\t{tmp}')
    return res

def get_lbls(lst_s: list[dict]) -> set:
    lbls = [
        i
        if i != 'MajorClaim' and i != 'Implicit Claim'
        else
        'Claim'
        for s in lst_s
        for i in s.get('label')
    ]
    return set(lbls)

'''
# def get_all_labels(data: dict) -> set:
#     labels = set()
#     for k,v in data.items():
#         labels = labels.union(v.get('list_label'))
#     return labels

# def get_nb_element_by_class(lbls: set, data: list[dict]) -> dict:
#     res = {}
#     for l in lbls:
#         tmp = [
#             x for x in data if l in x.get('label')
#         ]
#         res.update({l: len(tmp)})
#     return res

# def get_train_val_test_split(
#     data: list[dict],
#     lbls: set,
#     val_size=0.2,
#     test_size=0.2,
# ) -> tuple[list[dict], list[dict], list[dict]]:
#     test_sample = []
#     validation_sample = []
#     n_val = len(data)*val_size
#     n_test = len(data)*test_size
#     nb_elmt = get_nb_element_by_class(lbls, data)
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
#         s for s in data if s not in test_sample and s not in validation_sample
#     ]
#     return train_sample, validation_sample, test_sample
'''

def load_pe(path:str='./Data_jsonl/perssuasive_essays.jsonl') -> dict:
    all_data = []
    sentences = []
    with open(path, 'r') as f:
        for line in f:
            all_data.append(json.loads(line))
    for data in all_data:
        for arg in data.get('arguments'):
            tmp = {
                'text': data.get('text'),
                'sentences': arg.get('text'),
                'label': arg.get('type').split(',')
            }
            sentences.append(tmp)
    lbls_pe = get_lbls(sentences)
    train_set, validation_set, test_set = spl.get_train_val_test_split(
        data=sentences,
        lbls=lbls_pe
    )
    res = {
        'train': train_set,
        'validation': validation_set,
        'test': test_set,
        'list_label': lbls_pe
    }
    return res

def load_abstrct_neo(path:str='./Data_jsonl/abstrct_neoplasm.jsonl') -> dict:
    all_data = []
    sentences = []
    with open(path, 'r') as f:
        for line in f:
            all_data.append(json.loads(line))
    for data in all_data:
        for arg in data.get('arguments'):
            tmp = {
                'text': data.get('text'),
                'sentences': arg.get('text'),
                'label': arg.get('type').split(',')
            }
            sentences.append(tmp)
    lbls_abst = get_lbls(sentences)
    train_set, validation_set, test_set = spl.get_train_val_test_split(
        data=sentences,
        lbls=lbls_abst
    )
    res = {
        'train': train_set,
        'validation': validation_set,
        'test': test_set,
        'list_label': lbls_abst
    }
    return res

def load_abstrct_glau(
    path:str='./Data_jsonl/abstrct_glaucoma_test.jsonl'
) -> dict:
    all_data = []
    sentences = []
    with open(path, 'r') as f:
        for line in f:
            all_data.append(json.loads(line))
    for data in all_data:
        for arg in data.get('arguments'):
            tmp = {
                'text': data.get('text'),
                'sentences': arg.get('text'),
                'label': arg.get('type').split(','),
            }
            sentences.append(tmp)
    lbls_abst_glau = get_lbls(sentences)
    res = {
        'test': sentences,
        'list_label': lbls_abst_glau
    }
    return res

def load_abstrct_mixed(
    path:str='./Data_jsonl/abstrct_mixed_test.jsonl'
) -> dict:
    all_data = []
    sentences = []
    with open(path, 'r') as f:
        for line in f:
            all_data.append(json.loads(line))
    for data in all_data:
        for arg in data.get('arguments'):
            tmp = {
                'text': data.get('arguments'),
                'sentences': arg.get('text'),
                'label': arg.get('type').split(',')
            }
            sentences.append(tmp)
    lbls_abst_mixed = get_lbls(sentences)
    res = {
        'test': sentences,
        'list_label': lbls_abst_mixed,
    }
    return res

def load_mtp1(path:str='./Data_jsonl/microtext_p1.jsonl') -> dict:
    all_data = []
    sentences = []
    with open(path, 'r') as f:
        for line in f:
            all_data.append(json.loads(line))
    for data in all_data:
        text = get_mt(data.get('edu'))
        for edu in data.get('edu'):
            adu = map_edu_adu(edu, data.get('adu'))
            for i in adu:
                tmp = {
                    'topic': data.get('topic'),
                    'text': text,
                    'sentences': i.get('sentences'),
                    'label': i.get('label').split(',')
                }
                sentences.append(tmp)
    lbl_mtp1 = get_lbls(sentences)
    train_set, validation_set, test_set = spl.get_train_val_test_split(
        data=sentences,
        lbls=lbl_mtp1,
    )
    res = {
        'train': train_set,
        'validation': validation_set,
        'test': test_set,
        'list_label': lbl_mtp1
    }
    return res
        

def load_mtp2(path:str='./Data_jsonl/microtext_p2.jsonl') -> dict:
    all_data = []
    sentences = []
    with open(path, 'r') as f:
        for line in f:
            all_data.append(json.loads(line))
    for data in all_data:
        text = get_mt(data.get('edu'))
        for edu in data.get('edu'):
            adu = map_edu_adu(edu, data.get('adu'))
            for i in adu:
                tmp = {
                    'topic': data.get('topic'),
                    'text': text,
                    'sentences': i.get('sentences'),
                    'label': i.get('label').split(',')
                }
                sentences.append(tmp)
    lbl_mtp2 = get_lbls(sentences)
    train_set, validation_set, test_set = spl.get_train_val_test_split(
        data=sentences,
        lbls=lbl_mtp2,
    )
    res = {
        'train': train_set,
        'validation': validation_set,
        'test': test_set,
        'list_label': lbl_mtp2
    }
    return res

def load_all_datasets(paths: dict) -> tuple[dict, set]:
    res = {
        'mtp1': load_mtp1(paths.get('mtp1')),
        'mtp2': load_mtp2(paths.get('mtp2')),
        'pe': load_pe(paths.get('pe')),
        'abstrct': load_abstrct_neo(paths.get('abstrct')),
    }
    labels = spl.get_all_labels(res)
    return res, labels