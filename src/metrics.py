import pandas as pd
import numpy as np

def get_precision_recall(data):
    """Get the precision or the recall
    """
    try:
        tmp = data.apply(
            lambda x: x['pred'] in x['lbl'],
            axis=1
        )
        tp = tmp.value_counts().loc[True]
        score = tp / len(tmp)
        fp_fn = len(tmp) - tp
        return score, tp, fp_fn
    except (KeyError, ZeroDivisionError):
        tp = 0
        score = 0
        fp_fn = len(tmp) - tp
        return score, tp, fp_fn

def get_recall_multi(data, label):
    """Get the recall, used only for the FD task
    """
    n_lbl = len(label)
    ratio = []
    for l in label:
        c = data['pred'].value_counts()
        try:
            ratio.append(min((1/n_lbl) * (c[l] / (len(data)/n_lbl)), 1/n_lbl))
        except:
            ratio.append(min((1/n_lbl) * (0 / (len(data)/n_lbl)), 1/n_lbl))
    rec = sum(ratio)
    tp = np.round(len(data)*rec)
    fn = len(data) - tp
    return rec, tp, fn

def get_f1(precision, recall):
    """Get the F1 score
    """
    try:
        f1 = 2 * ((precision * recall) / (precision + recall))
    except ZeroDivisionError:
        f1 = 0
    return f1

def get_metrics_single(change_lbl, data, labels):
    """Get the different metrics (F1, precision and recall)
    """
    tp_preci = 0
    tp_rec = 0
    fn = 0
    fp = 0
    res = {}
    data['pred'] = data['pred'].str.lower()
    data['lbl'] = data['lbl'].apply(lambda x: [i.lower() for i in x])
    for l in labels:
        df_on_pred = data[data['pred'] == l[0].lower()]
        df_on_label = data.apply(
            lambda x: x if l[0].lower() in x['lbl'] else np.nan,
            result_type='broadcast',
            axis = 1
        ).dropna()
        precision, tp_p, fp_p = get_precision_recall(df_on_pred)
        recall, tp_r, fn_r = get_precision_recall(df_on_label)
        f1 = get_f1(precision, recall)
        n_lbl = change_lbl(l)
        res.update({str(n_lbl): (f1, precision, recall)})
        fn += fn_r
        fp += fp_p
        tp_preci += tp_p
        tp_rec += tp_r
    precision_all_data = tp_preci / (tp_preci + fp)
    recall_all_data = tp_rec / (tp_rec + fn)
    f1_all_data = get_f1(precision_all_data, recall_all_data)
    res.update({
        'score_all_data': (f1_all_data, precision_all_data, recall_all_data)
    })
    return res

def get_metrics_multi(change_lbl, data, labels):
    """Get the different metrics (F1, precision and recall), used only for the FD task
    """
    tp_preci = 0
    tp_rec = 0
    fn = 0
    fp = 0
    res = {}
    data['pred'] = data['pred'].str.lower()
    data['lbl'] = data['lbl'].apply(lambda x: [i.lower() for i in x])
    for l in labels:
        df_pred_lbl = data.apply(
            lambda x: x if x['lbl'] == [i.lower() for i in l] else np.nan,
            result_type='broadcast',
            axis=1
        ).dropna()
        precision, tp_p, fp_p = get_precision_recall(df_pred_lbl)
        recall, tp_r, fn_r = get_recall_multi(df_pred_lbl, l)
        f1 = get_f1(precision, recall)
        n_lbl = change_lbl(l)
        res.update({str(n_lbl): (f1, precision, recall)})
        fn += fn_r
        fp += fp_p
        tp_preci += tp_p
        tp_rec += tp_r
    precision_all_data = tp_preci / (tp_preci + fp)
    recall_all_data = tp_rec / (tp_rec + fn)
    f1_all_data = get_f1(precision_all_data, recall_all_data)
    res.update({
        'score_all_data': (f1_all_data, precision_all_data, recall_all_data)
    })
    return res

def get_metrics(change_lbl, data:pd.DataFrame, is_multi_lbl:bool=True):
    """Get the different metrics

    Parameters
    ----------
    change_lbl
        task specific function to unifie the name of the labels
    data : pd.DataFrame
        prediction made during testing
    is_multi_lbl : bool, optional
        set to False for all tasks except the FD task, by default True

    Returns
    -------
    dict
        dictionary containing the metrics
    dict
        used only for the FD task, dictionary containing the metrics
    """
    data_single = data.apply(
        lambda x: x if len(x['lbl'])<=1 else np.nan,
        axis=1,
        result_type='broadcast'
    ).dropna()
    data_multi = data.apply(
        lambda x: x if len(x['lbl'])>1 else np.nan,
        axis=1,
        result_type='broadcast'
    ).dropna()
    labels_single = data_single['lbl'].value_counts().index
    labels_multi = data_multi['lbl'].value_counts().index
    scores_single = get_metrics_single(change_lbl, data_single, labels_single)
    if is_multi_lbl:
        scores_multi = get_metrics_multi(change_lbl, data_multi, labels_multi)
    else:
        scores_multi = pd.DataFrame()
    return scores_single, scores_multi