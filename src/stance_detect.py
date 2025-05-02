import json
import itertools
import random
import pandas as pd
import numpy as np

import src.sampling as spl
import src.prompting as prt
import src.training as tr
import src.metrics as metrics
import src.plot as plot

from ast import literal_eval
from datasets import Dataset

def change_lbl(labels:list) -> list:
    return labels

def unifie_labels(label:str) -> str:
    if label == 'Pro'  or label == 'PRO' or label == 'support':
        return 'For'
    elif label == 'Con' or label == 'CON' or label == 'contest' or label == 'attack':
        return 'Against'
    else:
        return label

def map_aqm(labels:list, sentences:list) -> list:
    res = []
    for lbl in labels:
        idx = lbl.get('claim_idx')
        stance = lbl.get('stance')
        tmp = {
            'sentences': sentences[idx],
            'label': stance
        }
        res.append(tmp)
    return res

def get_claims(claims: list) -> list:
    res = []
    for claim in claims:
        tmp = {
            'sentences': claim.get('claimCorrectedText'),
            'label' : claim.get('stance')
        }
        res.append(tmp)
    return res

def load_ibm_claim_polarity(path:str) -> dict:
    all_data = []
    sentences = []
    with open(path, 'r') as f:
        for line in f:
            all_data.append(json.loads(line))
    for data in all_data:
        claims = get_claims(data.get('claims'))
        for claim in claims:
            tmp = {
                'topic': data.get('topicText'),
                'sentences': claim.get('sentences'),
                'label': unifie_labels(claim.get('label')).split(',')
            }
            sentences.append(tmp)
    lbl_ibm_claim_pola = spl.get_labels(sentences)
    train_set, validation_set, test_set = spl.get_train_val_test_split(
        data=sentences,
        lbls=lbl_ibm_claim_pola
    )
    res = {
        'train': train_set,
        'validation': validation_set,
        'test': test_set,
        'list_label': lbl_ibm_claim_pola
    }
    return res

def load_comarg(path:str) -> dict:
    all_data = []
    sentences = []
    with open(path, 'r') as f:
        for line in f:
            all_data.append(json.loads(line))
    for data in all_data:
        comment = data.get('comment')
        argument = data.get('argument')
        tmp = {
            'topic': data.get('topic'),
            'sentences': comment.get('text'),
            'label': unifie_labels(
                comment.get('stance')
            )
        }
        tmp1 = {
            'topic': data.get('topic'),
            'sentences': argument.get('text'),
            'label': unifie_labels(
                argument.get('stance')
            )
        }
        sentences.append(tmp)
        sentences.append(tmp1)
    sentences = [
        dict(t) for t in {tuple(s.items()) for s in sentences}
    ]
    for s in sentences:
        s.update({'label': s.get('label').split(',')})
    lbl_comarg = spl.get_labels(sentences)
    train_set, validation_set, test_set = spl.get_train_val_test_split(
        data=sentences,
        lbls=lbl_comarg
    )
    res = {
        'train': train_set,
        'validation': validation_set,
        'test': test_set,
        'list_label': lbl_comarg
    }
    return res

def load_nlas(path:str) -> dict:
    # TODO
    pass

def load_iam_stance(path:str) -> dict:
    all_data = []
    sentences = []
    with open(path, 'r') as f:
        for line in f:
            all_data.append(json.loads(line))
    for data in all_data:
        tmp = {
            'topic': data.get('topic'),
            'sentences': data.get('sentence'),
            'label': unifie_labels(data.get('label_stance')).split(',')
        }
        sentences.append(tmp)
    lbl_iam_stance = spl.get_labels(sentences)
    train_set, validation_set, test_set = spl.get_train_val_test_split(
        data=sentences,
        lbls=lbl_iam_stance
    )
    res = {
        'train': train_set,
        'validation': validation_set,
        'test': test_set,
        'list_label': lbl_iam_stance
    }
    return res

def load_fever(path:str) -> dict:
    all_data = []
    sentences = []
    with open(path, 'r') as f:
        for line in f:
            all_data.append(json.loads(line))
    for data in all_data:
        tmp = {
            'sentences': data.get('claim'),
            'label': unifie_labels(data.get('label')).split(',')
        }
        sentences.append(tmp)
    lbl_fever = spl.get_labels(sentences)
    train_set, validation_set, test_set = spl.get_train_val_test_split(
        data=sentences,
        lbls=lbl_fever
    )
    res = {
        'train': train_set,
        'validation': validation_set,
        'test': test_set,
        'list_label': lbl_fever
    }
    return res

def load_aqm(path:str) -> dict:
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
                'sentences': e.get('sentences'),
                'label': unifie_labels(e.get('label'))
            }
            sentences.append(tmp)
    sentences = [
        dict(t) for t in {tuple(s.items()) for s in sentences}
    ]
    for s in sentences:
        s.update({'label': s.get('label').split(',')})
    lbl_aqm = spl.get_labels(sentences)
    train_set, validation_set, test_set = spl.get_train_val_test_split(
        data=sentences,
        lbls=lbl_aqm
    )
    res = {
        'train': train_set,
        'validation': validation_set,
        'test': test_set,
        'list_label': lbl_aqm
    }
    return res

def load_all_datasets(paths:dict) -> tuple[dict, set]:
    res = {
        'ibm_claim_pola': load_ibm_claim_polarity(paths.get('ibm_claim_pola')),
        'comarg': load_comarg(paths.get('comarg')),
        'iam_stance': load_iam_stance(paths.get('iam_stance')),
        'aqm': load_aqm(paths.get('aqm'))
    }
    labels = spl.get_all_labels(res)
    return res, labels

def format_user_prompt(d:dict, labels:set) -> str:
    user_prt = ''
    topic = 'unknown'
    sentences = d.get('sentences')
    if 'topic' in d:
        topic = d.get('topic')
    user_prt = f'[TOPIC]: {topic}\n[SENTENCE]: {sentences}\n'
    return user_prt

def load_data(paths:dict, sys_prt:str, n_sample:int) -> tuple:
    data, labels = load_all_datasets(paths)
    spl_data = spl.get_all_spl(data, labels, n_sample)
    prt_train, prt_val, prt_test = prt.get_prt(
        format_user_prompt,
        data=spl_data,
        labels=labels,
        sys_prt=sys_prt
    )
    return labels, prt_train, prt_val, prt_test

def get_data(savefile:dict) -> tuple:
    converter = {'prompt': literal_eval, 'answer': literal_eval}
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
        title=f'Stance Detection: Scores {n_sample} sample',
        file_plot=savefile.get('plot_single'),
        file_metric=savefile.get('metric_single')
    )
    return model, tokenizer

def run_training_stance_detect(
    model,
    tokenizer,
    training_args,
    max_seq_length:int,
    n_sample:int,
    paths:dict,
    sys_prt:str,
    do_sample:bool,
    savefile:dict
):
    print(f'##### Load Data #####')
    if do_sample:
        labels, prt_train, prt_val, prt_test = load_data(
            paths=paths,
            sys_prt=sys_prt,
            n_sample=n_sample
        )
        prt_train.to_csv(savefile.get('train_spl_file'), index=False)
        prt_val.to_csv(savefile.get('val_spl_file'), index=False)
        prt_test.to_csv(savefile.get('test_spl_file'), index=False)
    else:
        labels, prt_train, prt_val, prt_test = get_data(savefile)
    plot.stat_sample(
        change_lbl,
        task_name='Stance Detection',
        sample_train=prt_train,
        sample_val=prt_val,
        sample_test=prt_test,
        labels=labels,
        savefile=savefile
    )
    data_train, data_val, data_test = prt.get_datasets(
        tokenizer=tokenizer,
        train=prt_train,
        val=prt_val,
        test=prt_test
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
    return m, t

    # print(f'##### Testing #####')
    # result_test = tr.test(
    #     model=model,
    #     tokenizer=tokenizer,
    #     data_test=data_test,
    #     labels=labels,
    #     result_file=savefile.get('test_result_file')
    # )
    # print(f'##### Metrics and plot #####')
    # metric, _ = metrics.get_metrics(change_lbl, result_test, is_multi_lbl=False)
    # plot.plot_metric(
    #     metric=metric,
    #     title=f'Stance Detection: Scores {n_sample} sample',
    #     file_plot=savefile.get('plot_single'),
    #     file_metric=savefile.get('metric_single')
    # )
    # return model, tokenizer