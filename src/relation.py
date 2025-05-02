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
    replace_lbl = {
        'add': 'support',
        'und': 'attack',
        'reb': 'attack',
        'supports': 'support',
        'attacks': 'attack',
        'Partial-Attack': 'attack',
        'Support': 'support',
        'Attack': 'attack',
        'no_relation': 'no relation',
        'CON': 'attack',
        'PRO': 'support',
        'Implicit Attack': 'attack',
        'Implicit Support': 'support',
        'No use': 'no relation',
        'sup': 'support'
    }
    if replace_lbl.get(label) is not None:
        return replace_lbl.get(label)
    else:
        return label

def get_claims(claims:list) -> list:
    res = []
    for claim in claims:
        tmp = {
            'sentences': claim.get('claimCorrectedText'),
            'label' : claim.get('stance')
        }
        res.append(tmp)
    return res

def get_args(id_arg:str, all_args:list) -> str:
    for arg in all_args:
        if arg.get('id') == id_arg:
            return arg.get('text')

def get_edu(id_edu:str, edu:list) -> str:
    for e in edu:
        for k,v in e.items():
            if k == id_edu:
                return v

def map_adu_edu(id_arg:str, adu:list, edu:list) -> str:
    for a in adu:
        for k,v in a.items():
            if v == id_arg:
                return get_edu(k, edu)

def get_args_mt(relation:dict, adu:list, edu:list, rel:list) -> str:
    if relation.get('trg')[0] != 'c':
        src = map_adu_edu(relation.get('src'), adu, edu)
        trg = map_adu_edu(relation.get('trg'), adu, edu)
        return src, trg
    else:
        for r in rel:
            if r.get('id') == relation.get('trg'):
                src = map_adu_edu(relation.get('src'), adu, edu)
                trg_rel_src = map_adu_edu(r.get('src'), adu, edu)
                trg_rel_trg = map_adu_edu(r.get('trg'), adu, edu)
                trg = f'{trg_rel_src} {trg_rel_trg}'
                return src, trg

def load_mtp1(path:str) -> dict:
    all_data = []
    sentences = []
    with open(path, 'r') as f:
        for line in f:
            all_data.append(json.loads(line))
    for data in all_data:
        edus = data.get('edu')
        adus = data.get('adu')
        rels = data.get('relations')
        for rel in rels:
            src, trg = get_args_mt(rel, adus, edus, rels)
            tmp = {
                'topic': data.get('topic'),
                'argument_src': src,
                'argument_trg': trg,
                'label': unifie_labels(rel.get('type')).split(',')
            }
            sentences.append(tmp)
    lbl_mtp1 = spl.get_labels(sentences)
    train_set, validation_set, test_set = spl.get_train_val_test_split(
        data=sentences,
        lbls=lbl_mtp1
    )
    res = {
        'train': train_set,
        'validation': validation_set,
        'test': test_set,
        'list_label': lbl_mtp1
    }
    return res

def load_mtp2(path:str) -> dict:
    all_data = []
    sentences = []
    with open(path, 'r') as f:
        for line in f:
            all_data.append(json.loads(line))
    for data in all_data:
        edus = data.get('edu')
        adus = data.get('adu')
        rels = data.get('relations')
        for rel in rels:
            src, trg = get_args_mt(rel, adus, edus, rels)
            tmp = {
                'topic': data.get('topic'),
                'argument_src': src,
                'argument_trg': trg,
                'label': unifie_labels(rel.get('type')).split(',')
            }
            sentences.append(tmp)
    lbl_mtp2 = spl.get_labels(sentences)
    train_set, validation_set, test_set = spl.get_train_val_test_split(
        data=sentences,
        lbls=lbl_mtp2
    )
    res = {
        'train': train_set,
        'validation': validation_set,
        'test': test_set,
        'list_label': lbl_mtp2
    }
    return res

def load_pe(path:str) -> dict:
    all_data = []
    sentences = []
    with open(path, 'r') as f:
        for line in f:
            all_data.append(json.loads(line))
    for data in all_data:
        all_args = data.get('arguments')
        for rel in data.get('relations'):
            arg_src = get_args(rel.get('arg_src'), all_args)
            arg_trg = get_args(rel.get('arg_trg'), all_args)
            tmp = {
                'argument_src': arg_src,
                'argument_trg': arg_trg,
                'label': unifie_labels(rel.get('type')).split(',')
            }
            sentences.append(tmp)
    lbl_pe = spl.get_labels(sentences)
    train_set, validation_set, test_set = spl.get_train_val_test_split(
        data=sentences,
        lbls=lbl_pe
    )
    res = {
        'train': train_set,
        'validation': validation_set,
        'test': test_set,
        'list_label': lbl_pe
    }
    return res

def load_abstrct(path:str) -> dict:
    all_data = []
    sentences = []
    with open(path, 'r') as f:
        for line in f:
            all_data.append(json.loads(line))
    for data in all_data:
        all_args = data.get('arguments')
        for rel in data.get('relations'):
            arg_src = get_args(rel.get('arg_src'), all_args)
            arg_trg = get_args(rel.get('arg_trg'), all_args)
            tmp = {
                'argument_src': arg_src,
                'argument_trg': arg_trg,
                'label': unifie_labels(rel.get('type')).split(',')
            }
            sentences.append(tmp)
    lbl_abstrct = spl.get_labels(sentences)
    train_set, validation_set, test_set = spl.get_train_val_test_split(
        data=sentences,
        lbls=lbl_abstrct
    )
    res = {
        'train': train_set,
        'validation': validation_set,
        'test': test_set,
        'list_label': lbl_abstrct
    }
    return res

def load_NK_debate(path:str) -> dict:
    all_data = []
    sentences = []
    with open(path, 'r') as f:
        for line in f:
            all_data.append(json.loads(line))
    for data in all_data:
        tmp = {
            'topic': data.get('topic'),
            'argument_src': data.get('argument2'),
            'argument_trg': data.get('argument1'),
            'label': unifie_labels(data.get('relation')).split(',')
        }
        sentences.append(tmp)
    lbl_nk_debate = spl.get_labels(sentences)
    train_set, validation_set, test_set = spl.get_train_val_test_split(
        data=sentences,
        lbls=lbl_nk_debate
    )
    res = {
        'train': train_set,
        'validation': validation_set,
        'test': test_set,
        'list_label': lbl_nk_debate
    }
    return res

def load_node_debate(path:str) -> dict:
    all_data = []
    sentences = []
    with open(path, 'r') as f:
        for line in f:
            all_data.append(json.loads(line))
    for data in all_data:
        tmp = {
            'topic': data.get('topic'),
            'argument_src': data.get('arg_src'),
            'argument_trg': data.get('arg_trg'),
            'label': unifie_labels(data.get('label')).split(',')
        }
        sentences.append(tmp)
    lbl_node_debate = spl.get_labels(sentences)
    train_set, validation_set, test_set = spl.get_train_val_test_split(
        data=sentences,
        lbls=lbl_node_debate
    )
    res = {
        'train': train_set,
        'validation': validation_set,
        'test': test_set,
        'list_label': lbl_node_debate
    }
    return res

def load_node_angrymen(path:str) -> dict:
    # TODO
    pass

def load_ibm_pola(path:str) -> dict:
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
                'argument_src': claim.get('sentences'),
                'argument_trg': data.get('topicText'),
                'label': unifie_labels(claim.get('label')).split(',')
            }
            sentences.append(tmp)
    lbl_ibm_pola = spl.get_labels(sentences)
    train_set, validation_set, test_set = spl.get_train_val_test_split(
        data=sentences,
        lbls=lbl_ibm_pola
    )
    res = {
        'train': train_set,
        'validation': validation_set,
        'test': test_set,
        'list_label': lbl_ibm_pola
    }
    return res

def load_comarg(path:str) -> dict:
    all_data = []
    sentences = []
    with open(path, 'r') as f:
        for line in f:
            all_data.append(json.loads(line))
    for data in all_data:
        tmp = {
            'topic': data.get('topic'),
            'argument_src': data.get('comment').get('text'),
            'argument_trg': data.get('argument').get('text'),
            'label': unifie_labels(data.get('label')).split(',')
        }
        sentences.append(tmp)
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

def load_all_datasets(paths:dict) -> dict:
    res = {
        'mtp1': load_mtp1(paths.get('mtp1')),
        'mtp2': load_mtp2(paths.get('mtp2')),
        'pe': load_pe(paths.get('pe')),
        'abstrct': load_abstrct(paths.get('abstrct')),
        'nk_debate': load_NK_debate(paths.get('nk_debate')),
        'node': load_node_debate(paths.get('node')),
        'ibm_claim_pola': load_ibm_pola(paths.get('ibm_claim_pola')),
        'comarg': load_comarg(paths.get('comarg')),
    }
    labels = spl.get_all_labels(res)
    return res, labels

def format_user_prompt(d:dict, labels:set) -> str:
    user_prt = ''
    topic = 'unknown'
    argument_src = d.get('argument_src')
    argument_trg = d.get('argument_trg')
    if 'topic' in d:
        topic = d.get('topic')
    user_prt = f'[RELATION]: {labels}\n[TOPIC]: {topic}\n[SOURCE]: {argument_src}\n[TARGET]: {argument_trg}\n'
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

def get_data(savefile):
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
        title=f'Relation Classification: Scores {n_sample} sample',
        file_plot=savefile.get('plot_single'),
        file_metric=savefile.get('metric_single')
    )
    return model, tokenizer

def run_training_relation(
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
        task_name='Relation Classification',
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
    #     title=f'Relation Classification: Scores {n_sample} sample',
    #     file_plot=savefile.get('plot_single'),
    #     file_metric=savefile.get('metric_single')
    # )
    # return model, tokenizer