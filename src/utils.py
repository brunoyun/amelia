from unsloth import FastLanguageModel
from unsloth import is_bfloat16_supported
from transformers import TrainingArguments

import json
import numpy as np

def get_savefile(
    task_name:str,
    spl_name:str,
    m_name:str,
    n_sample:int,
    epoch:int,
    train_resp:str
) -> dict:
    labels_file = f'./sampled_data/{task_name}/labels.csv'
    train_spl_file = f'./sampled_data/{task_name}/{spl_name}_train.csv'
    val_spl_file = f'./sampled_data/{task_name}/{spl_name}_val.csv'
    test_spl_file = f'./sampled_data/{task_name}/{spl_name}_test.csv'
    test_result_file = f'./test_res/{task_name}/test_res_{m_name}_{epoch}e{n_sample}{spl_name}{train_resp}.csv'
    file_stat_train = f'./img/{task_name}/{m_name}_{epoch}e{n_sample}{spl_name} _stat_train.png'
    file_stat_val = f'./img/{task_name}/{m_name}_{epoch}e{n_sample}{spl_name}_stat_val.png'
    file_stat_test = f'./img/{task_name}/{m_name}_{epoch}e{n_sample}{spl_name}_stat_test.png'
    file_plot_single = f'./img/{task_name}/{m_name}_{epoch}e{n_sample}{spl_name}{train_resp}_res_single.png'
    file_plot_multi = f'./img/{task_name}/{m_name}_{epoch}e{n_sample}{spl_name} {train_resp}_res_multi.png'
    file_metric_single = f'./test_res/{task_name}/{m_name}_{epoch}e{n_sample}{spl_name}{train_resp}_metric_single.csv'
    file_metric_multi = f'./test_res/{task_name}/{m_name}_{epoch}e{n_sample}{spl_name}{train_resp}_metric_multi.csv'
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
    }
    return d_file

def load_model(
    model_name:str,
    max_seq_length:int,
    dtype,
    load_in_4bit:bool,
    gpu_mem_use:float,
    epoch:float,
    outputs_dir:str,
    n_eval_step:int,
):
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
        r = 16, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
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
        report_to = "tensorboard", # Use this for WandB etc
        eval_strategy="steps",
        eval_steps=n_eval_step,
    )
    return model, tokenizer, training_args

def config_fallacies(
    model_name:str,
    max_seq_length:int,
    dtype,
    load_in_4bit:bool,
    gpu_mem_use:float,
    n_sample:int,
    epoch:int,
    n_eval:int,
    paths:dict,
    system_prompt:str,
    save_result:bool,
    do_sample:bool,
) -> dict:
    m_name = model_name.split('/')[1]
    task_name = 'fallacies'
    train_resp = '_train_resp'
    n_eval_step = np.floor((n_sample/32)/n_eval)
    spl_name = 'spl2'
    d_file = None
    if do_sample:
        spl_name = 'spl'
    outputs_dir = f'./outputs/{task_name}/{m_name}_{epoch}e{n_sample}{spl_name}{train_resp}'
    if save_result:
        d_file = get_savefile(
            task_name=task_name,
            spl_name=spl_name,
            m_name=m_name,
            n_sample=n_sample,
            epoch=epoch,
            train_resp=train_resp,
        )
    print(f'##### Load Model and Tokenizer #####')
    model, tokenizer, training_args = load_model(
        model_name=model_name,
        max_seq_length=max_seq_length,
        dtype=dtype,
        load_in_4bit=load_in_4bit,
        gpu_mem_use=gpu_mem_use,
        epoch=epoch,
        outputs_dir=outputs_dir,
        n_eval_step=n_eval_step
    )
    config = {
        'model': model,
        'tokenizer': tokenizer,
        'training_args': training_args,
        'max_seq_length': max_seq_length,
        'n_sample': n_sample,
        'spl_name': spl_name,
        'paths': paths,
        'sys_prt': system_prompt,
        'do_sample': do_sample,
        'savefile': d_file
    }
    return config

def config_aduc(
    model_name:str,
    max_seq_length:int,
    dtype,
    load_in_4bit:bool,
    gpu_mem_use:float,
    n_sample:int,
    epoch:int,
    n_eval:int,
    paths:dict,
    system_prompt:str,
    save_result:bool,
    do_sample:bool,
) -> dict:
    m_name = model_name.split(',')[1]
    task_name = 'aduc'
    train_resp = '_train_resp'
    n_eval_step = np.floor((n_sample/32)/n_eval)
    spl_name = 'spl2'
    d_file = None
    if do_sample:
        spl_name = 'spl'
    outputs_dir = f'./outputs/{task_name}/{m_name}_{epoch}e{n_sample}{spl_name}{train_resp}'
    if save_result:
        d_file = get_savefile(
            task_name=task_name,
            spl_name=spl_name,
            m_name=m_name,
            n_sample=n_sample,
            epoch=epoch,
            train_resp=train_resp,
        )
    print(f'#### Load Model and Tokenizer')
    model, tokenizer, training_args = load_model(
        model_name=model_name,
        max_seq_length=max_seq_length,
        dtype=dtype,
        load_in_4bit=load_in_4bit,
        gpu_mem_use=gpu_mem_use,
        epoch=epoch,
        outputs_dir=outputs_dir,
        n_eval_step=n_eval_step
    )
    config = {
        'model': model,
        'tokenizer': tokenizer,
        'training_args': training_args,
        'max_seq_length': max_seq_length,
        'n_sample': n_sample,
        'spl_name': spl_name,
        'paths': paths,
        'sys_prt': system_prompt,
        'do_sample': do_sample,
        'savefile': d_file
    }
    return config

def get_config(task:str=None)->dict:
    with open('./config.json', 'r') as conf_file:
        conf = json.loads(conf_file.read())
    config = {
        'fallacies': config_fallacies(**conf.get('fallacies')),
        'aduc': config_aduc(**conf.get('aduc'))
    }
    if task is not None and config.get(task) is not None:
        return config.get(task)
    else:
        print(f'Error while getting the configuration')