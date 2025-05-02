import json
import itertools
import pandas as pd

import src.metrics as metrics
import src.plot as plot
import src.sampling as spl
import src.prompting as prt
import src.training as tr


from ast import literal_eval
from datasets import Dataset

### Datasets function ###

def change_lbl(labels: list) -> list:
    new_lbl = []
    replace_lbl = {
        'appeal to nature': 'AN',
        'straw man': 'STM',
        'false dilemma': 'FD',
        'appeal to tradition': 'AT',
        'causal oversimplification': 'COS',
        'appeal to majority': 'AM',
        'ad hominem': 'AH',
        'appeal to ridicule': 'AR',
        'circular reasoning': 'CR',
        'false analogy': 'FA',
        'false causality': 'FC',
        'appeal to fear': 'AF',
        'appeal to worse problems': 'AWP',
        'none': 'NONE',
        'guilt by association': 'GA',
        'equivocation': 'EQ',
        'appeal to authority': 'AA',
        'hasty generalization': 'HG',
        'slippery slope': 'SS',
        'ad populum': 'AP'
    }
    for l in labels:
        new_lbl.append(replace_lbl.get(l))
    return new_lbl

def fix_labels_mafalda(d: dict) -> dict:
    """Change the name of the labels an element of the Mafalda Dataset:
    'nothing' -> 'none', 'appeal to (false) authority' -> 'appeal to authority'
    and remove the labels with less than 10 elements 

    Parameters
    ----------
    d : dict
        dictionary containing the sentence and the associated labels: 
        { 'sentences': '...', 'label': [ ... ] }

    Returns
    -------
    dict
        dictionary with the changed labels
    """
    lbl_to_remove = [
        'appeal to anger',
        'appeal to positive emotion',
        'tu quoque',
        'fallacy of division',
        'appeal to pity'
    ]
    to_replace = {
        'nothing': 'none',
        'appeal to (false) authority' : 'appeal to authority'
    }
    lbl_lst = list(itertools.chain.from_iterable(d.get('label')))
    tmp = [
        e
        for e in lbl_lst
        if (((len(lbl_lst) > 1 and e != 'nothing') or len(lbl_lst) == 1) and e not in lbl_to_remove)
    ]
    tmp = [to_replace.get(i, i) for i in tmp]
    d.update({'label': tmp})
    return d

def get_parent_comment(comments: list[dict], id: str) -> str:
    """Get the parent comment by id

    Parameters
    ----------
    comments : list[dict]
        list of the different comments
    id : str
        id of the parent comment

    Returns
    -------
    str
        text of the parent comment
    """
    for i in comments:
        if i.get('id') == id:
            return i.get('comment')

def load_cocolofa(path: str) -> dict:
    """Load the CoCoLoFa dataset

    Parameters
    ----------
    path : str, optional
        paht to the CoCoLoFa jsonl file, by default './Data_jsonl/cocolofa.jsonl'

    Returns
    -------
    dict
        dictionary containing the data split into 3 sets (train, validation, test):
        {
            'train': train_data_split,
            'validation': val_data_split,
            'test': test_data_split,
            'list_label': label of the dataset
        }
    """
    all_data = []
    sentences = []
    with open(path, 'r') as f:
        for line in f:
            all_data.append(json.loads(line))
    for i in all_data:
        for j in i.get('comments'):
            tmp = {}
            if j.get('respond_to') != '':
                parent_comment = get_parent_comment(
                    comments=i.get('comments'),
                    id=j.get('respond_to')
                )
            else:
                parent_comment = j.get('respond_to')
            tmp.update({
                'id': j.get('id'),
                'respond_to': parent_comment,
                'title': i.get('title'),
                'label': j.get('fallacy').split(','),
                'sentences': j.get('comment'),
            })
            sentences.append(tmp)
    lbls_cocolofa = spl.get_labels(sentences)
    train_set, validation_set, test_set = spl.get_train_val_test_split(
        data=sentences,
        lbls=lbls_cocolofa
    )
    res = {
        'train': train_set,
        'validation': validation_set,
        'test': test_set,
        'list_label': lbls_cocolofa
    }
    return res

def load_mafalfa(path: str) -> dict:
    """Load the MAFALDA dataset

    Parameters
    ----------
    path : str, optional
        path to the MAFALDA jsonl file, by default './Data_jsonl/mafalda.jsonl'

    Returns
    -------
    dict
        dictionary containing the data split into 3 sets (train, validation, test):
        {
            'train': train_data_split,
            'validation': val_data_split,
            'test': test_data_split,
            'list_label': label of the dataset
        }
    """
    all_data = []
    sentences = []
    with open(path, 'r') as f:
        for line in f:
            all_data.append(json.loads(line))
    for i in all_data:
        n_i = list(map(fix_labels_mafalda, i.get('sentences_with_labels')))
        if n_i != []:
            for j in n_i:
                if j.get('label') != []:
                    tmp = {}
                    tmp.update({
                        'sentences': j.get('sentences'),
                        'label': j.get('label'),
                        'text': i.get('text')
                    })
                    sentences.append(tmp)
    lbls_mafalda = spl.get_labels(sentences)
    train_set, validation_set, test_set = spl.get_train_val_test_split(
        data=sentences,
        lbls=lbls_mafalda
    )
    res = {
        'train' : train_set,
        'validation': validation_set,
        'test': test_set,
        'list_label': lbls_mafalda
    }
    return res

def load_all_datasets(paths: dict) -> tuple[dict, set]:
    """Load all datasets specified in the paths dictionary

    Parameters
    ----------
    paths : dict
        dictionary under the format { 'name_dataset': path to jsonl file of the dataset }

    Returns
    -------
    dict
        dictionary containing the data of each dataset under the following format: 
        {
            name_data : { 
                'train': ...,
                'val': ..., 
                'test': ..., 
                'list_label': set of label of the data 
            } 
        }
    set
        set of labels of the data
    """
    res = {
        'cocolofa': load_cocolofa(paths.get('cocolofa')),
        'mafalda': load_mafalfa(paths.get('mafalda'))
    }
    labels = spl.get_all_labels(res)
    # labels = get_lbls_inter(res)
    return res, labels

### Prompting Function ###

def format_user_prompt(d: dict, labels: set) -> str:
    """Format the user prompt

    Parameters
    ----------
    d : dict
        Element of the data to format
    labels : set
        set of labels of the data

    Returns
    -------
    str
        user prompt
    """
    user_prt = ''
    title = 'unknown'
    full_text = ''
    if 'title' in d:
        title = d.get('title')
    if 'text' in d:
        full_text = d.get('text')
    if 'respond_to' in d:
        full_text = f'{d.get("respond_to")} {d.get("sentences")}'
    sentences = f' {d.get("sentences")}'
    user_prt = f'[FALLACY]: {labels}\n[TITLE]: {title}\n[SENTENCE]: {sentences}\n[FULL TEXT]: {full_text}\n'
    return user_prt

# Run Training and Testing

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
    fallacies = set(
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
    return fallacies, prt_train, prt_val, prt_test

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
    metric_single, metric_multi = metrics.get_metrics(change_lbl, result_test)
    plot.plot_metric(
        metric=metric_single,
        title=f'Fallacies: Scores {n_sample} sample single label',
        file_plot=savefile.get('plot_single'),
        file_metric=savefile.get('metric_single')
    )
    plot.plot_metric(
        metric=metric_multi,
        title=f'Fallacies: Scores {n_sample} sample multi labels',
        file_plot=savefile.get('plot_multi'),
        file_metric=savefile.get('metric_multi')
    )
    return model, tokenizer

def run_training_fallacies(
    model,
    tokenizer,
    training_args,
    max_seq_length:int,
    n_sample:int,
    paths:dict,
    sys_prt:str,
    do_sample:bool,
    savefile:dict
) -> tuple:
    print(f'##### Load data #####')
    if do_sample:
        fallacies, prt_train, prt_val, prt_test = load_data(
            paths=paths,
            sys_prt=sys_prt,
            n_sample=n_sample
        )
        prt_train.to_csv(savefile.get('train_spl_file'), index=False)
        prt_val.to_csv(savefile.get('val_spl_file'), index=False)
        prt_test.to_csv(savefile.get('test_spl_file'), index=False)
    else:
        fallacies, prt_train, prt_val, prt_test = get_data(savefile)
    plot.stat_sample(
        change_lbl,
        task_name='Fallacies',
        sample_train=prt_train,
        sample_val=prt_val,
        sample_test=prt_test,
        labels=fallacies,
        savefile=savefile
    )
    data_train, data_val, data_test = prt.get_datasets(
        tokenizer=tokenizer, 
        train=prt_train,
        val=prt_val,
        test=prt_test
    )
    print('##### Training #####')
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
        labels=fallacies,
        n_sample=n_sample,
        savefile=savefile
    )
    return m, t

    # print(f'##### Testing #####')
    # result_test = tr.test(
    #     model=model,
    #     tokenizer=tokenizer,
    #     data_test=data_test,
    #     labels=fallacies,
    #     result_file=savefile.get('test_result_file')
    # )
    # print(f'##### Metrics and plot #####')
    # metric_single, metric_multi = metrics.get_metrics(change_lbl, result_test)
    # plot.plot_metric(
    #     metric=metric_single,
    #     title=f'Fallacies: Scores {n_sample} sample single label',
    #     file_plot=savefile.get('plot_single'),
    #     file_metric=savefile.get('metric_single')
    # )
    # plot.plot_metric(
    #     metric=metric_multi,
    #     title=f'Fallacies: Scores {n_sample} sample multi labels',
    #     file_plot=savefile.get('plot_multi'),
    #     file_metric=savefile.get('metric_multi')
    # )