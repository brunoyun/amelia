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
    return label

def get_quality(quality:dict) -> tuple[list, list]:
    qual_dim = []
    val = []
    for k,v in quality.items():
        if v.get('agg') != []:
            qual_dim.append(k)
            val.append(v.get('agg').split()[1].strip('()'))
    return qual_dim, val

def get_all_val(data:dict) -> list:
    res = []
    for k,v in data.get('label_quality').items():
        tmp = {
            'topic': data.get('topic'),
            'stance': data.get('stance'),
            'sentences': data.get('argument'),
            'qual_dim': k,
            'label': v.get('agg').split()[1].strip('()').split(',')
        }
        res.append(tmp)
    return res

def get_split(
    train:list[dict],
    val:list[dict],
    test:list[dict]
) -> tuple[list[dict], list[dict], list[dict]]:
    train_set, val_set, test_set = [], [], []
    for tr in train:
        for it in tr.get('all_val'):
            it['spl'] = tr['spl']
        train_set.extend(tr.get('all_val'))
    for v in val:
        for it in v.get('all_val'):
            it['spl'] = v['spl']
        val_set.extend(v.get('all_val'))
    for t in test:
        for it in t.get('all_val'):
            it['spl'] = t['spl']
        test_set.extend(t.get('all_val'))
    return train_set, val_set, test_set

def get_all_split(data:dict) -> dict:
    res = {}
    for k,v in data.items():
        train, val, test = get_split(v.get('train'), v.get('validation'), v.get('test'))
        res.update({
            k: {
                'train': train,
                'validation': val,
                'test': test,
            }
        })
    return res

def load_dagstuhl(path:str) -> dict:
    all_data = []
    sentences = []
    with open(path, 'r') as f:
        for line in f:
            all_data.append(json.loads(line))
    for data in all_data:
        qual_dim, val = get_quality(data.get('label_quality'))
        if qual_dim != [] and val != []:
            tmp = {
                'topic': data.get('topic'),
                'stance': data.get('stance'),
                'sentences': data.get('argument'),
                'qual_dim': qual_dim,
                'label': val,
                'all_val': get_all_val(data)
            }
            sentences.append(tmp)
    lbl_dag = spl.get_labels(sentences)
    train_set, validation_set, test_set = spl.get_train_val_test_split(
        data=sentences,
        lbls=lbl_dag
    )
    res = {
        'train': train_set,
        'validation': validation_set,
        'test': test_set,
        'list_label': lbl_dag
    }
    return res

def load_all_datasets(paths:dict) -> dict:
    res = {
        'dagsthul': load_dagstuhl(paths.get('dagsthul'))
    }
    labels = spl.get_all_labels(res)
    return res, labels

def format_user_prompt(d:dict, labels:set) -> str:
    qual_def = {
        'overall quality': 'Argumentation quality in total',
        'local_acceptability': 'A premise of an argument is acceptable if it is rationally worthy of being believed to be true',
        'appropriateness': "Argumentation has an appropriate style if the used language supports the creation of credibility and emotions as well as if it is proportional to the issue",
        'arrangement': "Argumentation is arranged properly if it presents the issue, the arguments, and its conclusion in the right order",
        'clarity': "Argumentation has a clear style if it uses correct and widely unambiguous language as well as if it avoids unnecessary complexity and deviation from the issue",
        'cogency': 'An argument is cogent if it has acceptable premises that are relevant to its conclusion and that are sufficient to draw the conclusion',
        'effectiveness': "Argumentation is effective if it persuades the target audience of (or corroborates agreement with) the author's stance on the issue",
        'global_acceptability': "Argumentation is acceptable if the target audience accepts both the consideration of the stated arguments for the issue and the way they are stated",
        'global_relevance': "Argumentation is relevant if it contributes to the issue's resolution, i.e., if it states arguments or other information that help to arrive at an ultimate conclusion",
        'global_sufficiency': "Argumentation is sufficient if it adequately rebuts those counter-arguments to it that can be anticipated",
        'reasonableness': "Argumentation is reasonable if it contributes to the issue's resolution in a sufficient way that is acceptable to the target audience.",
        'local_relevance': "A premise of an argument is relevant if it contributes to the acceptance or rejection of the argument's conclusion",
        'credibility': "Argumentation creates credibility if it conveys arguments and similar in a way that makes the author worthy of credence",
        'emotional_appeal': "Argumentation makes a successful emotional appeal if it creates emotions in a way that makes the target audience more open to the author's arguments",
        'sufficiency': "An argument's premises are sufficient if, together, they give enough support to make it rational to draw its conclusion",
    }
    user_prt = ''
    definition = qual_def.get(d.get('qual_dim'))
    topic = d.get('topic')
    stance = d.get('stance')
    sentences = d.get('sentences')
    user_prt = f'[QUALITY]: {labels}\n[TOPIC]: {topic}\n[STANCE]: {stance}\n[DEFINITION]: {d.get("qual_dim")}: {definition}\n[SENTENCE]: {sentences}\n'
    return user_prt

def load_data(paths:dict, sys_prt:str, n_sample:int) -> tuple:
    data, labels = load_all_datasets(paths)
    spl_data = spl.get_all_spl(data, labels, n_sample)
    ptr_train, prt_val, prt_test = prt.get_prt(
        format_user_prompt,
        data=spl_data,
        labels=labels,
        sys_prt=sys_prt
    )
    return labels, ptr_train, prt_val, prt_test

def get_data(savefile: dict) -> tuple:
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
    print(f'##### Metric and plot #####')
    metric, _ = metrics.get_metrics(change_lbl, result_test, is_multi_lbl=False)
    plot.plot_metric(
        metric=metric,
        title=f'Quality Assessment: Scores {n_sample} sample',
        file_plot=savefile.get('plot_single'),
        file_metric=savefile.get('metric_single')
    )
    return model, tokenizer

def run_training_quality(
    model,
    tokenizer,
    training_args,
    max_seq_length:int,
    n_sample:int,
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
            n_sample=n_sample
        )
        prt_train.to_csv(savefile.get('train_spl_file'), index=False)
        prt_val.to_csv(savefile.get('val_spl_file'), index=False)
        prt_test.to_csv(savefile.get('test_spl_file'), index=False)
    else:
        labels, prt_train, prt_val, prt_test = get_data(savefile)
    plot.stat_sample(
        change_lbl,
        task_name='Quality Assessment',
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
        training_args=training_args,
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