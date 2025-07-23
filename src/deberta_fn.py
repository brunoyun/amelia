import os
import re
import pandas as pd

from ast import literal_eval
from transformers import AutoTokenizer 
from transformers import AutoModelForSequenceClassification
from transformers import Trainer
from transformers import TrainingArguments
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, f1_score
from datasets import Dataset


def get_regex(task:str):
    regex  = {
        'aduc':  r'(?<=\[SENTENCE\]: )(.*?)(?=\n\[FULL TEXT\]:)',
        'claim_detection':  r'(?<=\[SENTENCE\]: )(.*?)(?=\n\[FULL TEXT\]:)',
        'evidence_detection': r'(?<=\[SENTENCE\]: )(.*)',
        'evidence_type': r'(?<=\[SENTENCE\]: )(.*)',
        'fallacies': r'(?s)(?<=\[SENTENCE\]: )(.*?)(?=\n\[FULL TEXT\]:)',
        'quality': r'(?<=\[SENTENCE\]: )(.*)',
        'relation': {
            'src': r'(?<=\[SOURCE\]: )(.*?)(?=\n\[TARGET\]:)',
            'trg': r'(?<=\[TARGET\]: )(.*)'
        },
        'stance_detection': r'(?<=\[SENTENCE\]: )(.*)',
    }
    return regex.get(task)

def parse_sentence(x, s):
    conv = x['conversations']
    match = re.search(s, conv[1].get('content'))
    if match:
        return match.group()
    else:
        print(f'Error: {match}, {conv[1].get("content")}, {s}')
        raise ValueError

def create_data_bert(task_name:str, path_src:str, path_trg:str):
    converter = {'conversations': literal_eval, 'answer': literal_eval}
    # s = r'^\./[^/]+/([^/]+)/'
    for root, dirs, files in os.walk(path_src):
        for file in files:
            if 'labels' not in file:
                f_path = os.path.join(root, file)
                df = pd.read_csv(
                    f_path,
                    converters=converter
                )
                task_regex = get_regex(task_name)
                if task_name != 'relation':
                    parse_data = df.apply(
                        lambda x: parse_sentence(x, task_regex),
                        axis=1,
                    )
                    df['conversations'] = parse_data
                else:
                    src_data = df.apply(
                        lambda x: parse_sentence(x, task_regex.get('src')),
                        axis=1,
                    )
                    trg_data = df.apply(
                        lambda x: parse_sentence(x, task_regex.get('trg')),
                        axis=1,
                    )
                    df['conversations'] = src_data
                    df['trg'] = trg_data
                df.to_csv(f'{path_trg}/{file}', index=False)

def parse_data(path:str):
    for root, dirs, files in os.walk(path):
        for dir in dirs:
            if 'mt_ft' not in dir:
                create_data_bert(dir, os.path.join(root,dir), f'spl_bert/{dir}')


def get_bert_data(task_name:str, tokenizer=None):
    train_data = pd.read_csv(f'./spl_bert/{task_name}/spl2_train.csv')
    test_data = pd.read_csv(f'./spl_bert/{task_name}/spl2_test.csv')
    if task_name != 'relation':
        train_data = train_data[['conversations', 'single_ans']]
        test_data = test_data[['conversations', 'single_ans']]
    else:
        train_data = train_data[['conversations', 'trg', 'single_ans']]
        test_data = test_data[['conversations','trg', 'single_ans']]
        tmp_train = train_data.apply(
            lambda x: f'{x["conversations"]} {tokenizer.sep_token} {x["trg"]}',
            axis=1,
        )
        tmp_test = test_data.apply(
            lambda x: f'{x["conversations"]} {tokenizer.sep_token} {x["trg"]}',
            axis=1,
        )
        train_data['conversations'] = tmp_train
        test_data['converstations'] = tmp_test
        train_data.drop(columns='trg', inplace=True)
        test_data.drop(columns='trg', inplace=True)
    train_data = train_data.rename(
        columns={'conversations': 'text', 'single_ans': 'label'}
    )
    test_data = test_data.rename(
        columns={'conversations': 'text', 'single_ans': 'label'}
    )
    return train_data, test_data

def tokenize_fn(batch, tokenizer):
    return tokenizer(
        batch['text'],
        padding='max_length',
        truncation=True,
        max_length=1024,
    )

def get_datasets(tokenizer, tokenize_fn, task_name:str):
    """Get the training and testing datasets
    """
    if task_name != 'relation':
        train_data, test_data = get_bert_data(task_name)
    else:
        train_data, test_data = get_bert_data(task_name, tokenizer)
    dataset = [
        Dataset.from_pandas(train_data),
        Dataset.from_pandas(test_data),
    ]
    for i, data in enumerate(dataset):
        data = data.map(lambda x: tokenize_fn(x, tokenizer), batched=True).class_encode_column(column='label')
        data = data.remove_columns(['text'])
        data.set_format('torch')
        dataset[i] = data
    train_dataset = dataset[0]
    test_dataset = dataset[1]
    return train_dataset, test_dataset

def compute_metrics(pred):
    """Compute the F1 score, precision on recall

    Parameters
    ----------
    pred : _type_
        prediction made by the model

    Returns
    -------
    dict
        dictionary containing the result
        {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'acc': acc,
        }
    """
    label = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(label, preds, average='weighted')
    acc = accuracy_score(label, preds)
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'acc': acc,
    }