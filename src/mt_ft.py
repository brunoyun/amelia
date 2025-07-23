from unsloth import apply_chat_template
from datasets import Dataset

import pandas as pd

import src.training as tr

from ast import literal_eval

def get_train_data(tokenizer, savefile: dict, chat_template:str) -> Dataset:
    """Get the Train data for Multi-task training

    Parameters
    ----------
    tokenizer
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

    Returns
    -------
    Dataset
        Train dataset
    """
    converter = {'conversations': literal_eval, 'answer': literal_eval}
    prt_train = pd.read_csv(
        savefile.get('train_spl_file'),
        converters=converter,
    )
    data_train = apply_chat_template(
        Dataset.from_pandas(prt_train),
        tokenizer=tokenizer,
        chat_template=chat_template,
        default_system_message= "",
    )
    # data_train = data_train[:30]
    # data_train = pd.DataFrame().from_records(data_train)
    # data_train = Dataset.from_pandas(data_train)
    return data_train

def run_mt_training(
    model,
    tokenizer,
    training_args,
    max_seq_length:int,
    savefile:dict,
    chat_template:str,
    save_model:bool,
    quantization:str,
):
    """Training function for Multi-task

    Parameters
    ----------
    model
    tokenizer
    training_args
        Training arguments
    max_seq_length : int
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
        set to True to save the model locally
    quantization : str
        gguf quantization, used only if save_model is set to True

    Returns
    -------
    Model trained in a Multi-task setting and tokenizer
    """
    print(f'##### Load Data #####')
    data_train = get_train_data(tokenizer, savefile, chat_template)
    print(f'##### Training #####')
    tr.train(
        model=model,
        tokenizer=tokenizer,
        data_train=data_train,
        data_val=None,
        max_seq_length=max_seq_length,
        training_args=training_args
    )
    if save_model:
        try:
            tr.save_model(
                directory=savefile.get('model_dir'),
                model=model,
                tokenizer=tokenizer,
                quantization=quantization,
            )
        except:
            return model, tokenizer
    return model, tokenizer
    
