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
    if label == 'Irrelevant Evidence':
        return 'Non-evidence'
    else:
        return label

def load_argsum_evi_cls(
    path:str,
    val_size:float=0.2,
    test_size:float=0.2
) -> dict:
    """Load the ArgSum dataset

    Parameters
    ----------
    path : str
        path to the dataset jsonl file
    val_size : float, optional
        fraction of the validation set, by default 0.2
    test_size : float, optional
        fraction of the test set, by default 0.2

    Returns
    -------
    dict
        dictionary containing the data split into 3 sets:
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
    for data in all_data:
        tmp = {
            'topic': data.get('topic'),
            'argument': data.get('argument'),
            'sentences': data.get('evidence'),
            'label': unifie_labels(data.get('label')).split(','),
        }
        sentences.append(tmp)
    lbl_argsum_evi = spl.get_labels(sentences)
    train_set, validation_set, test_set = spl.get_train_val_test_split(
        data=sentences,
        lbls=lbl_argsum_evi,
        val_size=val_size,
        test_size=test_size
    )
    res = {
        'train': train_set,
        'validation': validation_set,
        'test': test_set,
        'list_label': lbl_argsum_evi
    }
    return res

def load_iam_evi(path:str, val_size:float=0.2, test_size:float=0.2) -> dict:
    """Load the IAM dataset

    Parameters
    ----------
    path : str
        path to the dataset jsonl file
    val_size : float, optional
        fraction of the validation set, by default 0.2
    test_size : float, optional
        fraction of the test set, by default 0.2

    Returns
    -------
    dict
        dictionary containing the data split into 3 sets:
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
    for data in all_data:
        tmp = {
            'argument': data.get('claim'),
            'sentences': data.get('evidence'),
            'label': unifie_labels(data.get('label_evidence')).split(','),
        }
        sentences.append(tmp)
    lbl_iam_evi = spl.get_labels(sentences)
    train_set, validation_set, test_set = spl.get_train_val_test_split(
        data=sentences,
        lbls=lbl_iam_evi,
        val_size=val_size,
        test_size=test_size
    )
    res = {
        'train': train_set,
        'validation': validation_set,
        'test': test_set,
        'list_label': lbl_iam_evi
    }
    return res

def load_ibm_evi(path:str, val_size:float=0.2, test_size:float=0.2) -> dict:
    """Load the IBM Evidence dataset

    Parameters
    ----------
    path : str
        path to the dataset jsonl file
    val_size : float, optional
        fraction of the validation set, by default 0.2
    test_size : float, optional
        fraction of the test set, by default 0.2

    Returns
    -------
    dict
        dictionary containing the data split into 3 sets:
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
    for data in all_data:
        tmp = {
            'topic': data.get('topic'),
            'sentences': data.get('candidate'),
            'label': unifie_labels(data.get('label')).split(','),
        }
        sentences.append(tmp)
    lbl_ibm_evi = spl.get_labels(sentences)
    train_set, validation_set, test_set = spl.get_train_val_test_split(
        data=sentences,
        lbls=lbl_ibm_evi,
        val_size=val_size,
        test_size=test_size
    )
    res = {
        'train': train_set,
        'validation': validation_set,
        'test': test_set,
        'list_label': lbl_ibm_evi
    }
    return res

def load_all_datasets(
    paths:dict,
    val_size:float=0.2,
    test_size:float=0.2
) -> tuple[dict, set]:
    """Load all dataset for the ED task

    Parameters
    ----------
    paths : dict
        dictionary containing the path to the datasets files
    val_size : float, optional
        fraction of the validation set, by default 0.2
    test_size : float, optional
        fraction of the test set, by default 0.2

    Returns
    -------
    dict
        Dictionary containing the data from the different datasets
    set
        Labels for the ED task
    """
    res = {
        'argsum': load_argsum_evi_cls(paths.get('argsum'), val_size, test_size),
        'iam_evi': load_iam_evi(paths.get('iam_evi'), val_size, test_size),
        'ibm_evi': load_ibm_evi(paths.get('ibm_evi'), val_size, test_size)
    }
    labels = spl.get_all_labels(res)
    return res, labels

def format_user_prompt(d:dict, labels:set) -> str:
    """Create the user prompt used during training and inference

    Parameters
    ----------
    d : dict
        data
    labels : set
        labels for the task

    Returns
    -------
    str
        user prompt
    """
    user_prt = ''
    topic = 'unknown'
    sentences = d.get('sentences')
    if 'topic' in d:
        topic = d.get('topic')
    if 'argument' in d:
        argument = d.get('argument')
    else:
        argument = topic
    user_prt = f'[TOPIC]: {topic}\n[ARGUMENT]: {argument}\n[SENTENCE]: {sentences}\n'
    return user_prt

def load_data(
    paths:dict,
    sys_prt:str,
    n_sample:int,
    val_size:float=0.2,
    test_size:float=0.2
) -> tuple:
    """Load the data

    Parameters
    ----------
    paths : dict
        path to the datasets files
    sys_prt : str
        system prompt for the task
    n_sample : int
        number of training sample
    val_size : float, optional
        fraction of the validation set, by default 0.2
    test_size : float, optional
        fraction of the test set, by default 0.2

    Returns
    -------
    labels
        set of labels for the task
    ptr_train
        Train data
    prt_val
        Validation data
    prt_test
        Test data
    """
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
    """Get the sampled data from file

    Parameters
    ----------
    savefile : dict
        dictionary containing the file path to the sampled data
        {
            'labels_file': labels_file,
            'train_spl_file': train_spl_file,
            'val_spl_file': val_spl_file,
            'test_spl_file': test_spl_file,
            'test_result_file': test_result_file,
            'stat_train': file_stat_train,
            'stat_val': file_stat_val,
            'stat_test': file_stat_test,
            'plot_single': file_plot_single,
            'plot_multi': file_plot_multi,
            'metric_single': file_metric_single,
            'metric_multi': file_metric_multi,
            'outputs_dir': outputs_dir,
            'model_dir': model_dir
        }

    Returns
    -------
    labels
        set of labels for the task
    ptr_train
        Train data
    prt_val
        Validation data
    prt_test
        Test data
    """
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
    """Test the task after training

    Parameters
    ----------
    model
    tokenizer
    data_test : Dataset
        Test data
    labels : set
        set of labels for the task
    n_sample : int
        number of sample for the training
    savefile : dict
        dictionary containing the file path to the sampled data
        {
            'labels_file': labels_file,
            'train_spl_file': train_spl_file,
            'val_spl_file': val_spl_file,
            'test_spl_file': test_spl_file,
            'test_result_file': test_result_file,
            'stat_train': file_stat_train,
            'stat_val': file_stat_val,
            'stat_test': file_stat_test,
            'plot_single': file_plot_single,
            'plot_multi': file_plot_multi,
            'metric_single': file_metric_single,
            'metric_multi': file_metric_multi,
            'outputs_dir': outputs_dir,
            'model_dir': model_dir
        }

    Returns
    -------
    model and tokenizer
    """
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
        title=f'Evidence Detection: Scores {n_sample} sample',
        file_plot=savefile.get('plot_single'),
        file_metric=savefile.get('metric_single')
    )
    return model, tokenizer

def run_training_evidence_detect(
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
    """Training function for the ED task

    Parameters
    ----------
    model
    tokenizer
    training_args
    max_seq_length : int
    n_sample : int
        number of sample for training
    val_size : float
        fraction of the validation set
    test_size : float
        fraction of the test set
    paths : dict
        paths to the dataset file
    sys_prt : str
        system prompt for the task
    do_sample : bool
        re-sample the data if true
    savefile : dict
        dictionary containing the file path to the sampled data
        {
            'labels_file': labels_file,
            'train_spl_file': train_spl_file,
            'val_spl_file': val_spl_file,
            'test_spl_file': test_spl_file,
            'test_result_file': test_result_file,
            'stat_train': file_stat_train,
            'stat_val': file_stat_val,
            'stat_test': file_stat_test,
            'plot_single': file_plot_single,
            'plot_multi': file_plot_multi,
            'metric_single': file_metric_single,
            'metric_multi': file_metric_multi,
            'outputs_dir': outputs_dir,
            'model_dir': model_dir
        }
    chat_template : str
        chat template for the Llama model
    save_model : bool
        save the model locally if True
    quantization : str
        gguf quantization, used only if save_model=True

    Returns
    -------
    trained model and tokenizer
    """
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
            task_name='Evidence Detection',
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
            task_name='Evidence Detection',
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