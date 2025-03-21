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

    Args:
        d (dict): { 'sentences': '...', 'label': [ ... ] }

    Returns:
        dict: dictionnary with the changed labels
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

    Args:
        comments (list[dict]): list of comment
        id (str): id of the parent comment

    Returns:
        str: text of the comment
    """
    for i in comments:
        if i.get('id') == id:
            return i.get('comment')

def get_lbl_cocolofa(lst_s: list[dict]) -> set:
    """Get all the fallacy labels of the cocolofa Dataset

    Args:
        lst_s (list[dict]): all element of the dataset

    Returns:
        set: all the label present in the dataset
    """
    lbl_cocolofa = set()
    for s in lst_s:
        lbl_cocolofa.add(s.get('label'))
    return lbl_cocolofa

def get_lbl_mafalda(lst_s: list[dict]) -> set:
    """Get the labels of the Mafalda Dataset

    Args:
        lst_s (list[dict]): all element of the dataset

    Returns:
        set: all the label present in the dataset
    """
    lbl_mafalda = set()
    for s in lst_s:
        for i in s.get('label'):
            lbl_mafalda.add(i)
    return lbl_mafalda

def get_train_test_split(
    data: list[dict],
    lbls: set[str],
    test_size=0.2
) -> tuple[list[dict], list[dict]]:
    """Separate the data into train and test splits

    Args:
        data (list[dict]): all data
        lbls (set[str]): all the label
        test_size (float, optional): percentage of the data to use for the test split (value between 0.0 and 1.0). Defaults to 0.2.

    Returns:
        tuple[list[dict], list[dict]]: train_split, test_split
    """
    test_sample = []
    n = len(data)*test_size
    nb_elmt = get_nb_element_by_class(lbls, data)
    for l in lbls:
        ratio = nb_elmt.get(l) / sum(nb_elmt.values())
        n_sample = int(ratio*n)
        tmp = [x for x in data if l in x.get('label')]
        sample = random.sample(tmp, n_sample)
        test_sample.extend(sample)
    test_sample = random.sample(test_sample, len(test_sample))
    train_sample = [s for s in data if s not in test_sample]
    return train_sample, test_sample

def load_cocolofa(path: str='./Data_jsonl/cocolofa.jsonl') -> dict:
    """Load the CoCoLoFa dataset from file

    Args:
        path (str, optional): path to the jsonl file. Defaults to './Data_jsonl/cocolofa.jsonl'.

    Returns:
        dict: dictionary containing the data of the dataset :
        { 'train': train_split, 'test': test_split, 'list_label': all label of the dataset }
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
    train_set, test_set = get_train_test_split(sentences, lbls_cocolofa)
    res = {
        'train': train_set,
        'test': test_set,
        'list_label': lbls_cocolofa
    }
    return res

def load_mafalfa(path: str='./Data_jsonl/mafalda.jsonl') -> dict:
    """Load the MAFALDA dataset from file

    Args:
        path (str, optional): path to the jsonl file. Defaults to './Data_jsonl/mafalda.jsonl'.

    Returns:
        dict: dictionary containing the data of the dataset :
        { 'train': train_split, 'test': test_split, 'list_label': all label of the dataset }
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
    train_set, test_set = get_train_test_split(sentences, lbls_mafalda)
    res = {
        'train' : train_set,
        'test': test_set,
        'list_label': lbls_mafalda
    }
    return res

def load_all_dataset(paths: dict) -> dict:
    """Load all datasets specified in the paths dictionary

    Args:
        paths (dict): {
            'name of the dataset': 'path to jsonl file of the dataset'
        }

    Returns:
        dict: dictionary containing all the data
    """
    res = {
        'cocolofa': load_cocolofa(paths.get('cocolofa')),
        'mafalda': load_mafalfa(paths.get('mafalda'))
    }
    return res

### Sampling Function ###

def get_nb_element_by_class(lbls: set[str], data: list[dict]) -> dict:
    """Get the number of element of each class

    Args:
        lbls (set[str]): labels of each file
        data (list[dict]): data

    Returns:
        dict: _description_
    """
    res = {}
    for l in lbls:
        tmp = [
            x for x in data if l in x.get('label')
        ]
        res.update({l: len(tmp)})
    return res

def get_sample(
    data: list[dict],
    all_lbls: set,
    sample_per_class: dict
) -> list[dict]:
    """Sample a Dataset

    Args:
        data (list[dict]): data to sample
        all_lbls (set): labels of the dataset
        sample_per_class (dict): number of element to sample per class

    Returns:
        list[dict]: sampled_data 
    """
    res = []
    for l in all_lbls:
        tmp = [x for x in data if l in x.get('label')]
        if len(tmp) < sample_per_class.get(l):
            try:
                n_sample_length = sample_per_class.get(l) - len(tmp)
                sample = random.sample(tmp, len(tmp))
                over = random.sample(tmp, n_sample_length)
                sample.extend(over)
            except Exception:
                print(f'tmp: {tmp}')
                print(f'label: {l}')
                print(f'sample_per_class : {sample_per_class}')
                print(f'sample_lenght : {n_sample_length}')
        else:
            sample = random.sample(tmp, sample_per_class.get(l))
        res.extend(sample)
    return res

def get_nb_elmt_and_labels(data: dict) -> tuple[set, dict]:
    """Get the number of element of each class and all the labels from each dataset

    Args:
        data (dict): dict containing all the data to sample for the task

    Returns:
        tuple[set, dict]: set of all the labels and dictionary containing the number of element of each class
    """
    nb_elmt = {}
    all_lbls = set()
    for k,v in data.items():
        nb_elmt.update({
            k: get_nb_element_by_class(v.get('list_label'), v.get('train'))
        })
        all_lbls = all_lbls.union(v.get('list_label'))
    return all_lbls, nb_elmt

def get_ratio(x: pd.Series, n: float) -> pd.Series:
    """Get the sample ratio for a class in each dataset

    Args:
        x (pd.Series): number of element of a specific class
        n (float): number of element to sample in each class

    Returns:
        pd.Series: pandas Series containing the number of element to sample in a specific class
    """
    res = []
    lst = x.to_list()
    for i in range(len(lst)):
        res.append((lst[i] / x.sum())*n)
    return pd.Series(res, index=x.index)

def get_nb_sample_per_class(x: pd.Series) -> pd.Series:
    """Round the number of element to sample per class

    Args:
        x (pd.Series): number of element to sample in each dataset for a specific class

    Returns:
        pd.Series: pandas Series containing the round number of element to sample in each dataset for a specific class
    """
    res = []
    lst = x.to_list()
    for i in range(len(lst)):
        if lst[i] == x.min():
            res.append(int(np.ceil(lst[i])))
        else:
            res.append(int(lst[i]))
    return pd.Series(res, index=x.index)

def sample_data(data: dict, n_sample=300) -> dict:
    """Sample all the datasets for a specific task

    Args:
        data (dict): dictionary containing all the data
        n_sample (int, optional): number of element to sample across all the datasets. Defaults to 300.

    Returns:
        dict: dictionary containing the sampled data in each dataset:
        { 'name of the dataset' : data sample from the dataset }
    """
    sampled_data = {}
    all_lbls, nb_elmt = get_nb_elmt_and_labels(data)
    n = n_sample / len(all_lbls)
    df_elmt = pd.DataFrame().from_dict(nb_elmt)
    df_ratio = df_elmt.apply(lambda x: get_ratio(x, n), axis=1).fillna(0)
    df_sample_per_class = df_ratio.apply(get_nb_sample_per_class, axis=1)
    d = df_sample_per_class.T.to_dict('index')
    for k,v in data.items():
        sampled_data.update({k: get_sample(v.get('train'), all_lbls, d.get(k))})
    return sampled_data, all_lbls

### Prompting Function ###
# \nRespond in the following format with the explanation in the reasoning section and only the fallacy in the answer section:\n <reasoning>\n...\n</reasoning>\n<answer>\n...\n</answer>
SYSTEM_PROMPT = 'You are an expert in argumentation. Your task is to determine the type of fallacy in the given Sentence. The fallacy would be in the Fallacy Set. Utilize the Title and the Full Text as context to support your decision and provide an explanation of the reasoning behind your determination.'

def format_label(lbl: str | list) -> str:
    """Format the label to put as the answer

    Args:
        lbl (str | list): label of the element

    Returns:
        str: String representing the label
    """
    if isinstance(lbl, list):
        lbl = ', '.join(lbl)
    return lbl

def format_user_prompt(d: dict, labels: set) -> str:
    """Format the user prompt

    Args:
        d (dict): Element of the data to format

    Returns:
        str: String representation of the element to use in the prompt
    """
    user_prt = ''
    title = 'unknown'
    full_text = ''
    if 'title' in d:
        title = d.get('title')
    if 'text' in d:
        full_text = d.get('text')
    if 'respond_to' in d:
        full_text = f'{d.get('respond_to')} {d.get('sentences')}'
    sentences = f' {d.get('sentences')}'
    user_prt = f'Fallacy: {labels}\nTitle: {title}\nSentence: {sentences}\nFull text: {full_text}\n'
    return user_prt

def format_prompt(data: dict, labels: set) -> list[dict]:
    """Format the data into a prompt template

    Args:
        data (dict): all the data for the task

    Returns:
        list[dict]: list containing the prompt templates
    """
    res = []
    for k,v in data.items():
        for val in v:
            d = {
                'datasets': k,
                'prompt': [
                    {'role': 'system', 'content': SYSTEM_PROMPT},
                    {'role': 'user', 'content': format_user_prompt(val, labels)}
                ],
                'answer': format_label(val.get('label'))
            }
            res.append(d)
    return res

if __name__ == '__main__':
    pass