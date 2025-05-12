from unsloth import FastLanguageModel
import json
import itertools
import random
import re
import os
import time
import pandas as pd
import numpy as np

import src.sampling as spl
import src.prompting as prt
import src.training as tr
import src.metrics as metrics
import src.plot as plot

from ast import literal_eval
from datasets import Dataset

def change_lbl(labels: list) -> list:
    return labels

def unifie_labels(label:str) -> str:
    if label == 'Anecdotal Evidence' or label == 'Anecdotal Evidenc' or label == 'Case':
        return 'ANECDOTAL'
    elif label == 'Expert Evidence' or label == 'Expert Evidenc' or label == 'Expert':
        return 'EXPERT'
    elif label == 'Study Evidence' or label == 'Research':
        return 'STUDY'
    elif label == 'Others' or label == 'None Evidence':
        return 'NONE'
    elif label == 'Explanation':
        return 'EXPLANATION'
    else:
        return label

def map_aqm(labels:list, sentences:list) -> list:
    res = []
    for lbl in labels:
        idx_claim = lbl.get('claim_idx')
        idx_evidence = lbl.get('evidence_idx')
        evi_type = lbl.get('evidence_type')
        tmp = {
            'claim': sentences[idx_claim],
            'evidence': sentences[idx_evidence],
            'label': evi_type
        }
        res.append(tmp)
    return res

def load_ibm_type(path:str, val_size:float=0.2, test_size:float=0.2) -> dict:
    all_data = []
    sentences = []
    with open(path, 'r') as f:
        for line in f:
            all_data.append(json.loads(line))
    for data in all_data:
        tmp = {
            'topic': data.get('Topic'),
            'claim': data.get('Claim'),
            'evidence': data.get('CDE'),
            'label': unifie_labels(data.get('Type 1')).split(','),
        }
        sentences.append(tmp)
    lbl_ibm_type = spl.get_labels(sentences)
    train_set, validation_set, test_set = spl.get_train_val_test_split(
        data=sentences,
        lbls=lbl_ibm_type,
        val_size=val_size,
        test_size=test_size
    )
    res = {
        'train': train_set,
        'validation': validation_set,
        'test': test_set,
        'list_label': lbl_ibm_type
    }
    return res

def load_aqm(path:str, val_size:float=0.2, test_size:float=0.2) -> dict:
    all_data = []
    sentences = []
    with open(path, 'r') as f:
        for line in f:
            all_data.append(json.loads(line))
    for data in all_data:
        element = map_aqm(data.get('labels'), data.get('sents'))
        for e in element:
            tmp = {
                'topic': data.get('topic'),
                'claim': e.get('claim'),
                'evidence': e.get('evidence'),
                'label': unifie_labels(e.get('label')).split(',')
            }
            sentences.append(tmp)
    lbl_aqm = spl.get_labels(sentences)
    train_set, validation_set, test_set = spl.get_train_val_test_split(
        data=sentences,
        lbls=lbl_aqm,
        val_size=val_size,
        test_size=test_size
    )
    res = {
        'train': train_set,
        'validation': validation_set,
        'test': test_set,
        'list_label': lbl_aqm
    }
    return res

def load_argsum(path:str, val_size:float=0.2, test_size:float=0.2) -> dict:
    all_data = []
    sentences = []
    with open(path, 'r') as f:
        for line in f:
            all_data.append(json.loads(line))
    for data in all_data:
        if isinstance(data.get('type'), str):
            if len(data.get('type').split(',')) <= 1:
                tmp = {
                    'topic': data.get('topic'),
                    'evidence': data.get('evidence'),
                    'label': unifie_labels(data.get('type')).split(',')
                }
                sentences.append(tmp)
    lbl_argsum = spl.get_labels(sentences)
    train_set, validation_set, test_set = spl.get_train_val_test_split(
        data=sentences,
        lbls=lbl_argsum,
        val_size=val_size,
        test_size=test_size
    )
    res = {
        'train': train_set,
        'validation': validation_set,
        'test': test_set,
        'list_label': lbl_argsum
    }
    return res

def load_all_datasets(
    paths:dict,
    val_size:float=0.2,
    test_size:float=0.2
) -> tuple[dict, set]:
    res = {
        'ibm_type': load_ibm_type(paths.get('ibm_type'), val_size, test_size),
        'aqm': load_aqm(paths.get('aqm'), val_size, test_size),
        'argsum': load_argsum(paths.get('argsum'), val_size, test_size),
    }
    labels = spl.get_all_labels(res)
    return res, labels

def format_user_prompt(d:dict, labels:set) -> str:
    user_prt = ''
    claim = 'unknown'
    sentence = d.get('evidence')
    topic = d.get('topic')
    if 'claim' in d:
        claim = d.get('claim')
    user_prt = f'[TYPE]: {labels}\n[TOPIC]: {topic}\n[CLAIM]: {claim}\n[SENTENCE]: {sentence}\n'
    return user_prt

def load_data(
    paths:dict,
    sys_prt:str,
    n_sample:int,
    val_size:float=0.2,
    test_size:float=0.2
) -> tuple:
    data, labels = load_all_datasets(paths, val_size, test_size)
    spl_data = spl.get_all_spl(data, labels, n_sample)
    prt_train, prt_val, prt_test = prt.get_prt(
        format_user_prompt,
        data=spl_data,
        labels=labels,
        sys_prt=sys_prt
    )
    return labels, prt_train, prt_val, prt_test

def get_data(savefile:dict) -> tuple:
    converter = {'conversations': literal_eval, 'answer': literal_eval}
    labels = set(
        pd.read_csv(savefile.get('labels_file'))['labels'].tolist()
    )
    prt_train = pd.read_csv(
        savefile.get('train_spl_file'),
        converters=converter
    )
    prt_val = pd.read_csv(
        savefile.get('val_spl_file'),
        converters=converter
    )
    prt_test = pd.read_csv(
        savefile.get('test_spl_file'),
        converters=converter
    )
    return labels, prt_train, prt_val, prt_test

def test_task(
    model,
    tokenizer,
    data_test:Dataset,
    labels:set,
    n_sample:int,
    savefile:dict
) -> tuple:
    print(f'##### Testing #####')
    result_test = tr.test(
        model=model,
        tokenizer=tokenizer,
        data_test=data_test,
        labels=labels,
        result_file=savefile.get('test_result_file')
    )
    print(f'##### Metrics #####')
    metric, _ = metrics.get_metrics(change_lbl, result_test, is_multi_lbl=False)
    plot.plot_metric(
        metric=metric,
        title=f'Evidence Type: Scores {n_sample} sample',
        file_plot=savefile.get('plot_single'),
        file_metric=savefile.get('metric_single')
    )
    return model, tokenizer

def run_training_evidence_type(
    model,
    tokenizer,
    training_args,
    max_seq_length:int,
    n_sample:int,
    val_size:float,
    test_size:float,
    paths:dict,
    sys_prt:str,
    do_sample:bool,
    savefile:dict,
    chat_template:str,
    save_model: bool,
    quantization: str
):
    print(f'##### Load Data #####')
    if do_sample:
        labels, prt_train, prt_val, prt_test = load_data(
            paths=paths,
            sys_prt=sys_prt,
            n_sample=n_sample,
            val_size=val_size,
            test_size=test_size
        )
        prt_train.to_csv(savefile.get('train_spl_file'), index=False)
        prt_val.to_csv(savefile.get('val_spl_file'), index=False)
        prt_test.to_csv(savefile.get('test_spl_file'), index=False)
        plot.stat_sample(
            change_lbl,
            task_name='Evidence Type',
            sample_train=prt_train,
            sample_val=prt_val,
            sample_test=prt_test,
            labels=labels,
            savefile=savefile
        )
    else:
        labels, prt_train, prt_val, prt_test = get_data(savefile)
        plot.stat_sample(
            change_lbl,
            task_name='Evidence Type',
            sample_train=prt_train,
            sample_val=prt_val,
            sample_test=prt_test,
            labels=labels
        )
    data_train, data_val, data_test = prt.get_datasets(
        tokenizer=tokenizer,
        train=prt_train,
        val=prt_val,
        test=prt_test,
        chat_template=chat_template,
        sys_prt=sys_prt
    )
    print(f'##### Training #####')
    tr.train(
        model=model,
        tokenizer=tokenizer,
        data_train=data_train,
        data_val=data_val,
        max_seq_length=max_seq_length,
        training_args=training_args
    )
    m, t = test_task(
        model=model,
        tokenizer=tokenizer,
        data_test=data_test,
        labels=labels,
        n_sample=n_sample,
        savefile=savefile
    )
    if save_model:
        tr.save_model(savefile.get('model_dir'), m, t, quantization)
    return m, t 