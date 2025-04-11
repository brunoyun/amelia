import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def plot_stat_sample(
    change_lbl,
    sample: pd.DataFrame,
    lst_labels: set,
    title: str,
    savefile=None,
) -> None:
    width = 0.3
    fig, ax = plt.subplots()
    d = {}
    spl_len = {}
    spl_over = {}
    under = []
    over = []
    x = np.arange(len(lst_labels))
    b = 0
    bar_label = True
    dataset_names = sample['datasets'].value_counts().index.to_list()
    for name in dataset_names:
        spl_dataset = sample[sample['datasets'] == name]
        spl = spl_dataset[spl_dataset['spl'] == 'sample']
        over_spl = spl_dataset[spl_dataset['spl'] == 'oversample']
        len_s = {
            lbl: len(spl[spl['single_ans'] == lbl])
            for lbl in lst_labels
        }
            
        len_o = {
            lbl: len(over_spl[over_spl['single_ans'] == lbl])
            for lbl in lst_labels
        }
        spl_len.update({name: len_s})
        spl_over.update({name: len_o})
    df_spl_len = pd.DataFrame().from_dict(spl_len) # .sort_index()
    df_oversample = pd.DataFrame().from_dict(spl_over) # .sort_index()
    df_spl_len.index = change_lbl(df_spl_len.index)
    df_oversample.index = change_lbl(df_oversample.index)
    df_spl_len = df_spl_len.sort_index()
    df_oversample = df_oversample.sort_index()
    for name in dataset_names:
        under.append((name, df_spl_len[name]))
        over.append((name,df_oversample[name]))
    d.update({
        'under': under,
        'over': over,
    })
    for _,v in d.items():
        if bar_label:
            lbl = 'sample'
        else:
            lbl = 'oversample'
        p = ax.bar(x, v[0][1], width, label=f'{lbl} {v[0][0]}', bottom=b)
        # ax.bar_label(p, label_type='center')
        p = ax.bar(
            x=x,
            height=v[1][1],
            width=width,
            label=f'{lbl} {v[1][0]}',
            bottom=b + v[0][1]
        )
        # ax.bar_label(p, label_type='center')
        bar_label=False
        b = b + v[0][1] + v[1][1]
    # ax.set_yticks(np.arange(0, max(df_spl_len.max().values) + 2, step=1))
    lst_labels = change_lbl(lst_labels)
    ax.set_xticks(x, sorted(lst_labels), rotation=90)
    ax.legend(loc='best')
    ax.set_title(title)
    fig.set_size_inches(20, 10)
    # plt.ylim(0, max(df_spl_len.max().values) + 1)
    plt.show()
    if savefile is not None:
        fig.savefig(savefile, format='png')

def plot_metric(
    change_lbl,
    metric: dict,
    columns: list[str]=['f1', 'precision', 'recall'],
    title='',
    file_plot=None,
    file_metric=None,
):
    # rand_mark = pd.Series(np.full((len(metric),), 1/(len(metric)-1)))
    df_metric = pd.DataFrame().from_dict(
        metric,
        orient='index',
        columns=columns
    )
    fig, ax = plt.subplots(1, 1)
    df_metric.plot(
        ax=ax,
        kind='bar',
        figsize=(20,14),
        title=title,
    )
    # rand_mark.plot(ax=ax, color='red', linestyle='dashed')
    # ax.set_yticks(np.arange(0, 1.1, step=0.05))
    plt.xticks(rotation=90)
    plt.show()
    if file_plot is not None:
        fig.savefig(file_plot, format='png')
    if file_metric is not None:
        df_metric.to_csv(file_metric, header=['F1', 'Precision', 'Recall'])