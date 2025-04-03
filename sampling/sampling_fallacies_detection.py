import json
import itertools
import random
import pandas as pd
import numpy as np

### Datasets function ###

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

def get_lbl_cocolofa(lst_s: list[dict]) -> set:
    """Get the labels of the CoCoLoFa dataset

    Parameters
    ----------
    lst_s : list[dict]
        all element of the dataset

    Returns
    -------
    set
        set of labels in the CoCoLoFa dataset
    """
    lbl_cocolofa = set()
    for s in lst_s:
        lbl_cocolofa.add(s.get('label'))
    return lbl_cocolofa

def get_lbl_mafalda(lst_s: list[dict]) -> set:
    """Get the labels of the MAFALDA dataset

    Parameters
    ----------
    lst_s : list[dict]
        all element of the dataset

    Returns
    -------
    set
        set of labels in the MAFALDA dataset
    """
    lbl_mafalda = set()
    for s in lst_s:
        for i in s.get('label'):
            lbl_mafalda.add(i)
    return lbl_mafalda

def get_all_labels(data: dict) -> set:
    """Get all the labels

    Parameters
    ----------
    data : dict
        all the data

    Returns
    -------
    set
        set of labels of the data
    """
    labels = set()
    for k,v in data.items():
        labels = labels.union(v.get('list_label'))
    return labels

def get_lbls_inter(data: dict) -> set:
    """Get the labels present in all the different dataset

    Parameters
    ----------
    data : dict
        all the data

    Returns
    -------
    set
        set of labels present in all the different dataset
    """
    tmp = []
    labels = set()
    for k,v in data.items():
        tmp.append(v.get('list_label'))
    for i in tmp:
        if labels == set():
            for j in i:
                labels.add(j)
        else:
            labels.intersection(i)
    print(len(labels))
    return labels

def get_nb_element_by_class(lbls: set[str], data: list[dict]) -> dict:
    """Get the number of element of each class

    Parameters
    ----------
    lbls : set[str]
        set of labels of the data
    data : list[dict]
        data 

    Returns
    -------
    dict
        dictionary containing the number of element per class
    """
    res = {}
    for l in lbls:
        tmp = [
            x for x in data if l in x.get('label')
        ]
        res.update({l: len(tmp)})
    return res

def get_train_val_test_split(
    data: list[dict],
    lbls: set[str],
    val_size=0.2,
    test_size=0.2
) -> tuple[list[dict], list[dict], list[dict]]:
    """Separate the data into 3 distincts split

    Parameters
    ----------
    data : list[dict]
        data to split
    lbls : set[str]
        set of labels of the data
    val_size : float, optional
        ratio of the data to use for the validation split, by default 0.2
    test_size : float, optional
        ratio of the data to user for the test split, by default 0.2

    Returns
    -------
    list[dict]
        train split
    list[dict]
        validation split
    list[dict]
        test split
    """
    test_sample = []
    validation_sample = []
    n_val = len(data)*val_size
    n_test = len(data)*test_size
    nb_elmt = get_nb_element_by_class(lbls, data)
    for l in lbls:
        ratio = nb_elmt.get(l) / sum(nb_elmt.values())
        n_sample_val = int(ratio*n_val)
        n_sample_test = int(ratio*n_test)
        tmp = [x for x in data if l in x.get('label')]
        sample_val = random.sample(tmp, n_sample_val)
        tmp = [x for x in tmp if x not in sample_val]
        sample_test = random.sample(tmp, n_sample_test)
        validation_sample.extend(sample_val)
        test_sample.extend(sample_test)
    validation_sample = random.sample(validation_sample, len(validation_sample))
    test_sample = random.sample(test_sample, len(test_sample))
    train_sample = [
        s for s in data if s not in test_sample and s not in validation_sample
    ]
    return train_sample, validation_sample, test_sample

def load_cocolofa(path: str='./Data_jsonl/cocolofa.jsonl') -> dict:
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
            'val': val_data_split,
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
                'label': j.get('fallacy'),
                'sentences': j.get('comment'),
            })
            sentences.append(tmp)
    lbls_cocolofa = get_lbl_cocolofa(sentences)
    train_set, validation_set, test_set = get_train_val_test_split(
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

def load_mafalfa(path: str='./Data_jsonl/mafalda.jsonl') -> dict:
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
            'val': val_data_split,
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
    lbls_mafalda = get_lbl_mafalda(sentences)
    train_set, validation_set, test_set = get_train_val_test_split(
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

def load_all_dataset(paths: dict) -> tuple[dict, set]:
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
    labels = get_all_labels(res)
    # labels = get_lbls_inter(res)
    return res, labels

### Sampling Function ###

def get_nb_element(labels: list[str]) -> pd.Series:
    """Get the number of elements per labels

    Parameters
    ----------
    labels : list[str]
        list of labels

    Returns
    -------
    Series
        Series containing the number of element per labels
    """
    labels = list(itertools.chain.from_iterable(labels))
    df = pd.DataFrame(labels)
    return df.value_counts()
    

def get_spl_datasets(
    data: pd.DataFrame,
    labels: set,
    n_sample: int,
    nb_element: pd.Series
) -> tuple[pd.DataFrame, dict, dict]:
    """Get the sample of one dataset

    Parameters
    ----------
    data : pd.DataFrame
        data of one dataset
    labels : set
        set of labels in the data
    n_sample : int
        number of element to sample
    nb_element : pd.Series
        number of element to sample per labels

    Returns
    -------
    DataFrame
        DataFrame containing the sampled data from the dataset
    dict
        dictionary containing the number of sampled element of the dataset
    dict
        dictionary containing the number of oversampled element of the dataset
    """
    oversampled_len_lbls = {}
    sple_len_lbls = {}
    spl_lst = []
    for l in labels:
        # print(l)
        df_tmp = data.apply(
            lambda x: x if l in x['answer'] else np.nan,
            result_type='broadcast',
            axis=1
        ).dropna()
        # print(df_tmp['prompt'].value_counts().values)
        df_tmp = df_tmp.drop_duplicates(subset=['prompt'])
        # print(df_tmp['prompt'].value_counts().values)
        n_sample_label = int(
            np.round(len(df_tmp) / (nb_element.loc[l].iloc[0]) * n_sample)
        )
        oversampled_len_lbls.update({l: 0})
        sple_len_lbls.update({l: n_sample_label})
        try:
            df_spl = df_tmp.sample(n=n_sample_label)
        except ValueError:
            df_spl = df_tmp.sample(n=n_sample_label, replace=True)
            oversampled_len_lbls.update({l: int(n_sample_label - len(df_tmp))})
            sple_len_lbls.update({l: len(df_tmp)})
        spl_lst.append(df_spl)
    df_spl = pd.concat(spl_lst)
    return df_spl, sple_len_lbls, oversampled_len_lbls

def get_spl(
    data: pd.DataFrame,
    labels: set,
    n_sample: int = 300
) -> tuple[pd.DataFrame, dict, dict]:
    """Get a sample of the data

    Parameters
    ----------
    data : DataFrame
        data to sample
    labels : set
        set of labels in the data
    n_sample : int, optional
        number of element to sample in the data, by default 300

    Returns
    -------
    DataFrame
        DataFrame containing the sampled data
    dict
        dictionary containing the number of sample per labels
    dict
        dictionary containing the number of oversampled element per labels
    """
    lst_spl = []
    oversampled_len_lbls = {}
    sple_len_lbls = {}
    nb_spl = int(n_sample / len(labels))
    nb_element = get_nb_element(data['answer'].to_list())
    names_dataset = data['datasets'].value_counts().index.to_list()
    for n in names_dataset:
        df = data.loc[data['datasets'] == n]
        df_spl, spl_len, oversample = get_spl_datasets(
            data=df,
            labels=labels,
            n_sample=nb_spl,
            nb_element=nb_element
        )
        oversampled_len_lbls.update({n: oversample})
        sple_len_lbls.update({n: spl_len})
        lst_spl.append(df_spl)
    df_res = pd.concat(lst_spl)
    return df_res, sple_len_lbls ,oversampled_len_lbls

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

def get_prt_train(
    data: list[dict],
    names: str,
    labels: set,
    system_prompt: str
) -> list[dict]:
    """Get the prompts for the train split

    Parameters
    ----------
    data : list[dict]
        data of the train split
    names : str
        name of the dataset
    labels : set
        set of label in the data
    system_prompt : str
        system prompt for the task

    Returns
    -------
    list[dict]
        list of prompts for the train split
    """
    prt = []
    for d in data:
        if isinstance(d.get('label'), list):
            for i in d.get('label'):
                p = {
                    'datasets': names,
                    'prompt': [
                        {'role': 'system', 'content': system_prompt},
                        {'role': 'user',
                         'content': format_user_prompt(d, labels)},
                        {'role': 'assistant', 'content': 
                            f'<|ANSWER|>{i}<|ANSWER|>.'}
                    ],
                    'answer': d.get('label')
                }
                prt.append(p)
        else:
            p = {
                'datasets': names,
                'prompt': [
                    {'role': 'system', 'content': system_prompt},
                    {'role': 'user', 'content': format_user_prompt(d, labels)},
                    {'role': 'assistant',
                     'content': f'<|ANSWER|>{d.get("label")}<|ANSWER|>.'}
                ],
                'answer': d.get('label').split(',')
            }
            prt.append(p)
    return prt

def get_prt_val_test(
    data: list[dict],
    names: str,
    labels: set,
    system_prompt: str
) -> list[dict]:
    """Get the prompts for the validation and test splits

    Parameters
    ----------
    data : list[dict]
        data of the split
    names : str
        name of the datasets
    labels : set
        set of labels in the data
    system_prompt : str
        system prompt for the task

    Returns
    -------
    list[dict]
        list prompts for the split
    """
    prt = []
    for d in data:
        if isinstance(d.get('label'), list):
            for i in d.get('label'):
                p = {
                    'datasets': names,
                    'prompt': [
                        {'role': 'system', 'content': system_prompt},
                        {'role': 'user',
                         'content': format_user_prompt(d, labels)},
                    ],
                    'answer': d.get('label')
                }
                prt.append(p)
        else:
            p = {
                'datasets': names,
                'prompt': [
                    {'role': 'system', 'content': system_prompt},
                    {'role': 'user', 'content': format_user_prompt(d, labels)},
                ],
                'answer': d.get('label').split(',')
            }
            prt.append(p)
    return prt

def get_prt(
    data: dict,
    labels: set,
    sys_prt: str
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Get the prompt of each data split

    Parameters
    ----------
    data : dict
        dictionary containing the different split of the data under the 
        following format:
        {
            name_data : { 
                'train': ...,
                'val': ..., 
                'test': ..., 
                'list_label': set of label of the data 
            } 
        }
    labels : set
        set of labels present in the data
    sys_prt : str
        System prompt to use for the task

    Returns
    -------
    DataFrame
        prompts of the train split
    DataFrame
        prompt of the validation split
    DataFrame
        prompt of the test split
    """
    prt_train = []
    prt_val = []
    prt_test = []
    for k,v in data.items():
        prt_train.extend(get_prt_train(v.get('train'), k, labels, sys_prt))
        prt_val.extend(
            get_prt_val_test(
                data=v.get('validation'),
                names=k,
                labels=labels,
                system_prompt=sys_prt
            )
        )
        prt_test.extend(get_prt_val_test(v.get('test'), k, labels, sys_prt))
    data_train = pd.DataFrame().from_records(prt_train)
    data_val = pd.DataFrame().from_records(prt_val)
    data_test = pd.DataFrame().from_records(prt_test)
    return data_train, data_val, data_test