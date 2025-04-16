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

# The label 'Other' indicates that the edu/adu is neither a Claim or a Premises

def change_lbl(labels: list) -> list:
    return labels

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

def unifie_labels(label: str) -> str:
    if label == 'MajorClaim' or label == 'Implicit Claim':
        return 'Claim'
    else:
        return label

def load_pe(path:str) -> dict:
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
                'label': unifie_labels(arg.get('type')).split(',')
            }
            sentences.append(tmp)
    lbls_pe = spl.get_labels(sentences)
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

def load_abstrct_neo(path:str) -> dict:
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
                'label': unifie_labels(arg.get('type')).split(',')
            }
            sentences.append(tmp)
    lbls_abst = spl.get_labels(sentences)
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
                'label': unifie_labels(arg.get('type')).split(','),
            }
            sentences.append(tmp)
    lbls_abst_glau = spl.get_labels(sentences)
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
                'label': unifie_labels(arg.get('type')).split(',')
            }
            sentences.append(tmp)
    lbls_abst_mixed = spl.get_labels(sentences)
    res = {
        'test': sentences,
        'list_label': lbls_abst_mixed,
    }
    return res

def load_mtp1(path:str) -> dict:
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
                    'label': unifie_labels(i.get('label')).split(',')
                }
                sentences.append(tmp)
    lbl_mtp1 = spl.get_labels(sentences)
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
        

def load_mtp2(path:str) -> dict:
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
                if i.get('label') != 'Other':
                    tmp = {
                        'topic': data.get('topic'),
                        'text': text,
                        'sentences': i.get('sentences'),
                        'label': unifie_labels(i.get('label')).split(',')
                    }
                    sentences.append(tmp)
    lbl_mtp2 = spl.get_labels(sentences)
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

def format_user_prompt(d:dict, labels:set) -> str:
    user_prt = ''
    topic = 'unknown'
    full_text = d.get('text')
    sentences = d.get('sentences')
    if 'topic' in d:
        topic = d.get('topic')
    user_prt = f'[TOPIC]: {topic}\n[SENTENCE]: {sentences}\n[FULL TEXT]: {full_text}\n'
    return user_prt

def run_aduc(
    model,
    tokenizer,
    training_args,
    max_seq_length:int,
    n_sample:int,
    spl_name:str,
    paths:dict,
    sys_prt:str,
    do_sample:bool,
    savefile:dict
):
    print(f'##### Load Data #####')
    if do_sample:
        data, labels = load_all_datasets(paths)
        spl_data = spl.get_all_spl(data, labels, n_sample)
        prt_train, prt_val, prt_test = prt.get_prt(
            format_user_prompt,
            data=spl_data,
            labels=labels,
            sys_prt=sys_prt
        )
        prt_train.to_csv(savefile.get('train_spl_file'), index=False)
        prt_val.to_csv(savefile.get('val_spl_file'), index=False)
        prt_test.to_csv(savefile.get('test_spl_file'), index=False)
    else:
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
    print(f'##### Testing #####')
    result_test = tr.test(
        model=model,
        tokenizer=tokenizer,
        data_test=data_test,
        labels=labels,
        result_file=savefile.get('test_result_file')
    )
    print(f'##### Metrics and plot #####')
    metric, _ = metrics.get_metrics(change_lbl, result_test, is_multi_lbl=False)
    plot.plot_stat_sample(
        change_lbl,
        sample=prt_train,
        lst_labels=labels,
        savefile=savefile.get('stat_train'),
        title=f'aduc: sample {spl_name} train'
    )
    plot.plot_stat_sample(
        change_lbl,
        sample=prt_val,
        lst_labels=labels,
        savefile=savefile.get('stat_val'),
        title=f'aduc: sample {spl_name} val'
    )
    plot.plot_stat_sample(
        change_lbl,
        sample=prt_test,
        labels=labels,
        savefile=savefile.get('stat_test'),
        title=f'aduc: sample {spl_name} test'
    )
    plot.plot_metric(
        metric=metric,
        title=f'aduc: Scores {n_sample} sample',
        file_plot=savefile.get('plot_single'),
        file_metric=savefile.get('metric_single')
    )