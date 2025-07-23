from unsloth import FastLanguageModel
from unsloth import is_bfloat16_supported
from transformers import TrainingArguments

import json
import datetime
import numpy as np

from src.fallacies import run_training_fallacies
from src.aduc import run_training_aduc
from src.claim_detect import run_training_claim_detect
from src.evidence_detect import run_training_evidence_detect
from src.stance_detect import run_training_stance_detect
from src.evidence_type import run_training_evidence_type
from src.relation import run_training_relation
from src.quality import run_training_quality
from src.mt_ft import run_mt_training

def get_savefile(
    task_name:str,
    spl_name:str,
    m_name:str,
    n_sample:int,
    epoch:int,
    train_resp:str,
    outputs_dir:str,
    time:str
) -> dict:
    """Get the filepath to the sampled data, plot and test result

    Parameters
    ----------
    task_name : str
        task name among [aduc, claim_detection, evidence_detection, evidence_type, fallacies, relation, stance_detection, mt_ft]
    spl_name : str
        name of the sample
    m_name : str
        model name
    n_sample : int
        number of element in the train sample
    epoch : int
        number of epochs
    train_resp : str
        
    outputs_dir : str
        path to the outputs directory
    time : str
        time stamp

    Returns
    -------
    dict
        dictionary containing all the file path 
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
    """
    labels_file = f'./sampled_data/{task_name}/labels.csv'
    train_spl_file = f'./sampled_data/{task_name}/{spl_name}_train.csv'
    val_spl_file = f'./sampled_data/{task_name}/{spl_name}_val.csv'
    test_spl_file = f'./sampled_data/{task_name}/{spl_name}_test.csv'
    test_result_file = f'./test_res/{task_name}/{time}_test_res_{m_name}_{epoch}e{n_sample}{spl_name}{train_resp}.csv'
    file_stat_train = f'./img/{task_name}/{n_sample}spl_{spl_name}_stat_train.png'
    file_stat_val = f'./img/{task_name}/{n_sample}spl_{spl_name}_stat_val.png'
    file_stat_test = f'./img/{task_name}/{n_sample}spl_{spl_name}_stat_test.png'
    file_plot_single = f'./img/{task_name}/{time}_{m_name}_{epoch}e{n_sample}{spl_name}{train_resp}_res_single.png'
    file_plot_multi = f'./img/{task_name}/{time}_{m_name}_{epoch}e{n_sample}{spl_name}{train_resp}_res_multi.png'
    file_metric_single = f'./test_res/{task_name}/{time}_{m_name}_{epoch}e{n_sample}{spl_name}{train_resp}_metric_single.csv'
    file_metric_multi = f'./test_res/{task_name}/{time}_{m_name}_{epoch}e{n_sample}{spl_name}{train_resp}_metric_multi.csv'
    model_dir = f'./gguf_model/{task_name}'
    d_file = {
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
    return d_file

def get_templates() -> str:
    """Get the Llama model chat template
    
    Returns
    -------
    str
        chat template of the Llama model
    """
    chat_template = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{SYSTEM}<|eot_id|><|start_header_id|>user<|end_header_id|>

{INPUT}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

{OUTPUT}<|eot_id|>"""
    return chat_template
    

def load_model(
    model_name:str,
    max_seq_length:int,
    dtype,
    load_in_4bit:bool,
    gpu_mem_use:float,
    epoch:float,
    outputs_dir:str,
    save_steps:int | None=None,
    n_eval_step:int | None=None,
    r_lora:int=16,
):
    """Load the model to train

    Parameters
    ----------
    model_name : str
    max_seq_length : int
    dtype : any
    load_in_4bit : bool
    gpu_mem_use : float
        fraction of the  gpu memory to use for the model, between 0.1 and 0.9
    epoch : float
        number of training epochs
    outputs_dir : str
        path to outputs directory
    save_steps : int | None, optional
        number of steps between saving to outputs dirs, by default None
    n_eval_step : int | None, optional
        number of steps between validation, by default None
    r_lora : int, optional
        Lora parameters, by default 16

    Returns
    -------
    model, tokenizer and training arguments
    """
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        dtype=dtype,
        load_in_4bit=load_in_4bit,
        fast_inference=True,
        gpu_memory_utilization=gpu_mem_use
    )
    model = FastLanguageModel.get_peft_model(
        model=model,
        r = r_lora, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj",],
        lora_alpha = 16,
        lora_dropout = 0, # Supports any, but = 0 is optimized
        bias = "none",    # Supports any, but = "none" is optimized
        # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
        use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long   context
        random_state = 3407,
        use_rslora = False,  # We support rank stabilized LoRA
        loftq_config = None, # And LoftQ
    )
    training_args = TrainingArguments(
        per_device_train_batch_size = 4, #2
        per_device_eval_batch_size= 4,
        gradient_accumulation_steps = 8, #4
        eval_accumulation_steps= 8,
        warmup_steps = 5,
        num_train_epochs = epoch, # Set this for 1 full training run.
        # max_steps = 60,
        learning_rate = 2e-4,
        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = outputs_dir,
        save_strategy='steps',
        save_steps=save_steps if save_steps != None else 500,
        report_to = "tensorboard", # Use this for WandB etc
        eval_strategy="steps" if n_eval_step != None else "no",
        eval_steps=n_eval_step if n_eval_step != None else None,
    )
    return model, tokenizer, training_args

def load_training_config(
    task_name:str,
    model_name:str,
    paths:dict,
    system_prompt:str,
    max_seq_length:int=2048,
    dtype=None,
    load_in_4bit:bool=True,
    gpu_mem_use:float=0.6,
    r_lora:int=16,
    n_sample:int=4000,
    epoch:int=2,
    n_eval:int=8,
    val_size:float=0.2,
    test_size:float=0.2,
    do_sample:bool=True,
    save_model:bool=True,
    quantization:str | None=None
) -> dict:
    """Load the training configurations

    Parameters
    ----------
    task_name : str
        task name among [aduc, claim_detection, evidence_detection, evidence_type, fallacies, relation, stance_detection, mt_ft]
    model_name : str
        model name
    paths : dict
        path to the different datasets to use for training
    system_prompt : str
        system prompt associated to the task
    max_seq_length : int, optional
        by default 2048
    dtype : _type_, optional
        by default None
    load_in_4bit : bool, optional
        by default True
    gpu_mem_use : float, optional
        fraction of the gpu memory to use, by default 0.6
    r_lora : int, optional
        Lora parameters, by default 16
    n_sample : int, optional
        number of sample in the train set, by default 4000
    epoch : int, optional
        number of training epochs, by default 2
    n_eval : int, optional
        by default 8
    val_size : float, optional
        fraction of the validation set, by default 0.2
    test_size : float, optional
        fraction of the test set, by default 0.2
    do_sample : bool, optional
        redo a new sampling, by default True
    save_model : bool, optional
        save the model locally after training, by default True
    quantization : str | None, optional
        gguf quantization, by default None

    Returns
    -------
    dict
        dictionary containing the training configuration
        {
            'model': model,
            'tokenizer': tokenizer,
            'training_args': training_args,
            'max_seq_length': max_seq_length,
            'n_sample': n_sample,
            'val_size': val_size,
            'test_size': test_size,
            'paths': paths,
            'sys_prt': system_prompt,
            'do_sample': do_sample,
            'savefile': d_file,
            'chat_template': get_templates(),
            'save_model': save_model,
            'quantization': quantization
        }
    """
    m_name = model_name.split('/')[1]
    train_resp = '_train_resp'
    if n_eval != 0:
        n_eval_step = np.floor((n_sample/32)/n_eval)
        save_steps = np.round((n_sample/32)/2)
    spl_name = 'spl2'
    d_file = None
    time = datetime.datetime.now().strftime('%Y-%m-%d_%H_%M_%S')
    if do_sample:
        spl_name = f'{time}_spl'
    outputs_dir = f'./outputs/{task_name}/{time}_{m_name}_{epoch}e{n_sample}{spl_name}{train_resp}'
    d_file = get_savefile(
        task_name=task_name,
        spl_name=spl_name,
        m_name=m_name,
        n_sample=n_sample,
        epoch=epoch,
        train_resp=train_resp,
        outputs_dir=outputs_dir,
        time=time
    )
    print(f'##### Load Model and Tokenizer #####')
    if task_name != 'mt_ft':
        model, tokenizer, training_args = load_model(
            model_name=model_name,
            max_seq_length=max_seq_length,
            dtype=dtype,
            load_in_4bit=load_in_4bit,
            gpu_mem_use=gpu_mem_use,
            epoch=epoch,
            outputs_dir=outputs_dir,
            save_steps=save_steps,
            n_eval_step=n_eval_step,
            r_lora=r_lora
        )
        config = {
            'model': model,
            'tokenizer': tokenizer,
            'training_args': training_args,
            'max_seq_length': max_seq_length,
            'n_sample': n_sample,
            'val_size': val_size,
            'test_size': test_size,
            'paths': paths,
            'sys_prt': system_prompt,
            'do_sample': do_sample,
            'savefile': d_file,
            'chat_template': get_templates(),
            'save_model': save_model,
            'quantization': quantization
        }
    else:
        model, tokenizer, training_args = load_model(
            model_name=model_name,
            max_seq_length=max_seq_length,
            dtype=dtype,
            load_in_4bit=load_in_4bit,
            gpu_mem_use=gpu_mem_use,
            epoch=epoch,
            outputs_dir=outputs_dir,
            r_lora=r_lora
        )
        config={
            'model': model,
            'tokenizer': tokenizer,
            'training_args': training_args,
            'max_seq_length': max_seq_length,
            'savefile': d_file,
            'chat_template': get_templates(),
            'save_model': save_model,
            'quantization': quantization
        }
    return config

def fn_config(
    task_name:str,
    training_params:dict,
) -> dict:
    """Get the training configuration for a specific task

    Parameters
    ----------
    task_name : str
        task name among [aduc, claim_detection, evidence_detection, evidence_type, fallacies, relation, stance_detection, mt_ft]
    training_params : dict
        training parameters for the task

    Returns
    -------
    dict
        training configuration for the specified task
    """
    config = load_training_config(task_name=task_name, **training_params)
    return config

def get_config(task:str=None) -> dict:
    """Load the config.json file

    Parameters
    ----------
    task : str, optional
        task name among [aduc, claim_detection, evidence_detection, evidence_type, fallacies, relation, stance_detection, mt_ft], by default None

    Returns
    -------
    dict
        training configuration for the specified task
    """
    with open('./config.json', 'r') as conf_file:
        conf = json.loads(conf_file.read())
    return fn_config(task_name=task, **conf.get(task))

def run_training(task: str=None):
    """Run the training for a specific task

    Parameters
    ----------
    task : str, optional
        task name among [aduc, claim_detection, evidence_detection, evidence_type, fallacies, relation, stance_detection, mt_ft], by default None

    Returns
    -------
    trained model and tokenizer
    """
    if task is not None:
        config = get_config(task=task)
        match task:
            case 'fallacies':
                model, tokenizer = run_training_fallacies(**config)
            case 'aduc':
                model, tokenizer = run_training_aduc(**config)
            case 'claim_detection':
                model, tokenizer = run_training_claim_detect(**config)
            case 'evidence_detection':
                model, tokenizer = run_training_evidence_detect(**config)
            case 'stance_detection':
                model, tokenizer = run_training_stance_detect(**config)
            case 'evidence_type':
                model, tokenizer = run_training_evidence_type(**config)
            case 'relation':
                model, tokenizer = run_training_relation(**config)
            case 'quality':
                model, tokenizer = run_training_quality(**config)
            case 'mt_ft':
                model, tokenizer = run_mt_training(**config)
        return model, tokenizer
    else:
        print(f'Error while getting config for task {task}')