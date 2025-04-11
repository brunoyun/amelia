from datasets import Dataset
import pandas as pd

def formatting_prompt(tokenizer, data: dict):
    text = tokenizer.apply_chat_template(data.get('prompt'),tokenize = False, add_generation_prompt = False)
    return { 'text': text, }

def get_prt_train_val(
    format_user_prompt,
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
                    'spl': d.get('spl'),
                    'single_ans': i,
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
                'spl': d.get('spl'),
                'single_ans': d.get('label'),
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

def get_prt_test(
    format_user_prompt,
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
                    'spl': d.get('spl'),
                    'single_ans': i,
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
                'spl': d.get('spl'),
                'single_ans': d.get('label'),
                'prompt': [
                    {'role': 'system', 'content': system_prompt},
                    {'role': 'user', 'content': format_user_prompt(d, labels)},
                ],
                'answer': d.get('label').split(',')
            }
            prt.append(p)
    return prt

def get_prt(
    format_user_prompt,
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
        prt_train.extend(
            get_prt_train_val(
                format_user_prompt,
                data=v.get('train'),
                names=k,
                labels=labels, 
                system_prompt=sys_prt
            )
        )
        prt_val.extend(
            get_prt_train_val(
                format_user_prompt,
                data=v.get('validation'),
                names=k,
                labels=labels,
                system_prompt=sys_prt
            )
        )
        prt_test.extend(
            get_prt_test(
                format_user_prompt,
                data=v.get('test'),
                names=k,
                labels=labels,
                system_prompt=sys_prt
            )
        )
    data_train = pd.DataFrame().from_records(prt_train)
    data_val = pd.DataFrame().from_records(prt_val)
    data_test = pd.DataFrame().from_records(prt_test)
    return data_train, data_val, data_test

def get_datasets(
    tokenizer,
    train:pd.DataFrame,
    val:pd.DataFrame,
    test:pd.DataFrame
)->tuple[Dataset, Dataset, Dataset]:
    data_train = Dataset.from_pandas(train).map(
        lambda x: formatting_prompt(tokenizer, x),
        batched=True
    )
    data_val = Dataset.from_pandas(val).map(
        lambda x: formatting_prompt(tokenizer, x),
        batched=True
    )
    data_test = Dataset.from_pandas(test).map(
        lambda x: formatting_prompt(tokenizer, x),
        batched=True
    ).shuffle(seed=0)
    return data_train, data_val, data_test