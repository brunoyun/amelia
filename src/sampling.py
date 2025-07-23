import random
import itertools
import pandas as pd
import numpy as np

def get_labels(lst_s: list[dict]) -> set:
    lbl = [
        i
        for s in lst_s
        for i in s.get('label')
    ]
    return set(lbl)

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
    c_val = pd.DataFrame(labels).value_counts()
    c_val = c_val.rename_axis(None, axis=0)
    idx=[i[0] for i in c_val.index]
    c_val = pd.Series(c_val.values, index=idx).sort_index()
    return c_val

def get_nb_element_split(
    data: list[str],
    nb_element: pd.Series,
) -> pd.Series:
    """Get the number of element to sample for one split

    Parameters
    ----------
    data : list[str]
        data to sample
    nb_element : pd.Series

    Returns
    -------
    pd.Series
        number of element in one split
    """
    df = pd.DataFrame(data)
    df_elmt = get_nb_element(
        df['label'].apply(
            lambda x: x if isinstance(x, list) else x.split(',')
        )
    )
    nb_element = (nb_element + df_elmt).fillna(df_elmt).fillna(nb_element)
    return nb_element

def get_nb_element_all_split(
    data: dict, 
    labels: set
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Get the number for element in each split

    Parameters
    ----------
    data : dict
        data to sample
    labels : set
        set of the different labels

    Returns
    -------
    pd.Series
        number of element to sample for the train set
    pd.Series
        number of element to sample for the validation set
    pd.Series
        number of element to sample for the test set
    """
    nb_elmt_train = pd.Series(0, index=labels).sort_index()
    nb_elmt_val = pd.Series(0, index=labels).sort_index()
    nb_elmt_test = pd.Series(0, index=labels).sort_index()
    for k,v in data.items():
        nb_elmt_train = get_nb_element_split(
            data=v.get('train'),
            nb_element=nb_elmt_train,
        )
        nb_elmt_val = get_nb_element_split(
            data=v.get('validation'),
            nb_element=nb_elmt_val,
        )
        nb_elmt_test = get_nb_element_split(
            data=v.get('test'),
            nb_element=nb_elmt_test,
        )
    return nb_elmt_train, nb_elmt_val, nb_elmt_test

def get_spl_datasets(
    data: pd.DataFrame,
    labels: set,
    n_sample: int,
    nb_element: pd.Series,
    task:str=None
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
    task : str
        used only for the quality task, by default None

    Returns
    -------
    DataFrame
        DataFrame containing the sampled data from the dataset
    dict
        dictionary containing the number of sampled element of the dataset
    dict
        dictionary containing the number of oversampled element of the dataset
    """
    spl_lst = []
    for l in labels:
        df_tmp = data.apply(
            lambda x: x if l in x['label'] else np.nan,
            result_type='broadcast',
            axis=1
        ).dropna()
        if task == 'quality':
            all_val = []
            df_all_v = data.apply(
                lambda x: all_val.extend(x['all_val']),
                axis=1
            )
            df_all = pd.DataFrame().from_records(all_val)
            tmp = df_all.apply(
                lambda x: x if l in x['label'] else np.nan,
                result_type='broadcast',
                axis=1
            ).dropna()
            n_sample_label = int(
                np.round(len(tmp) / (nb_element.loc[l]) * n_sample)
            )
        else:
            n_sample_label = int(
                np.round(len(df_tmp) / (nb_element.loc[l]) * n_sample)
            )
        if (len(df_tmp) - n_sample_label) >= 0:
            df_spl = df_tmp.sample(n=n_sample_label)
            df_over = df_tmp.sample(n=0)
            df_spl['spl'] = 'sample'
        else:
            df_spl = df_tmp.sample(n=len(df_tmp))
            df_over = df_tmp.sample(
                n=n_sample_label - len(df_tmp),
                replace=True
            )
            df_spl['spl'] = 'sample'
            df_over['spl'] = 'oversample'
        spl_lst.append(df_spl)
        spl_lst.append(df_over)
    df_spl = pd.concat(spl_lst)
    return df_spl

def get_all_spl(
    data: dict,
    labels: set,
    n_sample: int=300,
    val_size: float=0.2,
    test_size: float=0.2,
    task:str=None
) -> dict:
    """Get the sampled data for training, validation and test set

    Parameters
    ----------
    data : dict
        Data to sample
    labels : set
        set of the different labels
    n_sample : int, optional
        number of element of the training sample, by default 300
    val_size : float, optional
        fraction of the validation set, by default 0.2
    test_size : float, optional
        fraction of the test set, by default 0.2
    task : str, optional
        used only for the quality task, by default None

    Returns
    -------
    dict
        dictionary containing the sampled data under the format
        {
            train
            validation
            test
        }
    """
    res = {}
    lst_spl_train, lst_spl_val, lst_spl_test = [], [], []
    nb_elmt_train, nb_elmt_val, nb_elmt_test = get_nb_element_all_split(
        data=data,
        labels=labels
    )
    nb_spl_tr = int(n_sample / len(labels))
    nb_spl_val = int((n_sample*val_size) / len(labels))
    nb_spl_test = int((n_sample*test_size) / len(labels))
    for k,v in data.items():
        spl_train = get_spl_datasets(
            data=pd.DataFrame(v.get('train')),
            labels=labels,
            n_sample=nb_spl_tr,
            nb_element=nb_elmt_train,
            task=task
        )
        spl_val = get_spl_datasets(
            data=pd.DataFrame(v.get('validation')),
            labels=labels,
            n_sample=nb_spl_val,
            nb_element=nb_elmt_val,
            task=task
        )
        spl_test = get_spl_datasets(
            data=pd.DataFrame(v.get('test')),
            labels=labels,
            n_sample=nb_spl_test,
            nb_element=nb_elmt_test,
            task=task
        )
        lst_spl_train.append(spl_train)
        lst_spl_val.append(spl_val)
        lst_spl_test.append(spl_test)
        res.update({
            k: {
                'train': spl_train.to_dict(orient='records'),
                'validation': spl_val.to_dict(orient='records'),
                'test': spl_test.to_dict(orient='records')
            }
        })
    return res