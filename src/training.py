from unsloth import FastLanguageModel
from unsloth import unsloth_train
from unsloth.chat_templates import train_on_responses_only

from datasets import Dataset
from transformers import TextStreamer
from trl import SFTTrainer

import itertools
import re
import pandas as pd

def gen(p, model, tokenizer, text_streamer):
    """Generation an answer following a prompt
    """
    txt = tokenizer.apply_chat_template(
        p,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to('cuda')
    output = model.generate(
        txt,
        streamer=text_streamer,
        max_new_tokens=128,
        pad_token_id=tokenizer.eos_token_id
    )
    return output

def format_output(answer: list, labels: set, few_shot:bool=False) -> list:
    """Formate the output of the model

    Parameters
    ----------
    answer : list
        answer of the model
    labels : set
        label of a specific task
    few_shot : bool, optional
        set to true if performing few-shot, by default False

    Returns
    -------
    list
        formate output
    """
    # s = '<[|]ANSWER[|]>'
    s = r'<\|ANSWER\|>(.*?)<\|(ANSWER|eot_id)\|>'
    labels = {lbl.lower() for lbl in labels}
    tmp = re.split(s, answer[0])
    pred = [
        re.sub(r'[^\w\s_-]','',i)
        for i in tmp
        if re.sub(r'[^\w\s_-]','',i).lower() in labels
    ]
    if few_shot:
        if len(pred) <= len(labels):
            pred = []
        else:
            print(f'## Few-shot : {pred} ##')
            pred = [pred[-1]]
    print(f'## Prediction : {pred} ##')
    return pred

def zero_shot_gen(
    data:Dataset,
    model,
    tokenizer,
    labels:set,
    text_streamer: TextStreamer,
    few_shot:bool=False,
) -> list:
    """Generate an answer

    Parameters
    ----------
    data : Dataset
        data
    model
    tokenizer
    labels : set
        labels of a specific task
    text_streamer : TextStreamer
    few_shot : bool, optional
        set to True if performing few-shot, by default False

    Returns
    -------
    list
        output of the generation
    """
    res= []
    prompt = data['conversations']
    for prt in prompt:
        out = gen(prt, model, tokenizer, text_streamer)
        decoded_out = tokenizer.batch_decode(out)
        pred = format_output(decoded_out, labels, few_shot)
        res.append(pred)
    return res

def train(
    model,
    tokenizer,
    data_train,
    data_val,
    max_seq_length,
    training_args
):
    """Train the model

    Parameters
    ----------
    model
    tokenizer
    data_train : Dataset
        Train data
    data_val : Dataset
        Validation data
    max_seq_length
    training_args
        Training arguments
    """
    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = data_train,
        eval_dataset=data_val,
        # formatting_func=formatting_prompt,
        dataset_text_field = "text",
        max_seq_length = max_seq_length,
        dataset_num_proc = 2,
        packing = False, # Can make training 5x faster for short sequences.
        args = training_args,
    )
    trainer = train_on_responses_only(
        trainer,
        # instruction_part="<|start_header_id|>user<|end_header_id|>\n\n",
        # response_part="<|start_header_id|>assistant<|end_header_id|>\n\n"
    )
    unsloth_train(trainer)
    
def test(
    model,
    tokenizer,
    data_test,
    labels,
    result_file=None,
    few_shot:bool=False
) -> pd.DataFrame:
    """Test the model

    Parameters
    ----------
    model
    tokenizer
    data_test : Dataset
        Test Dataset
    labels : set
        labels of a specific task
    result_file : optional
        filepath to save the test result, by default None
    few_shot : bool, optional
        set to True if performing few-shot, by default False

    Returns
    -------
    pd.DataFrame
        Test result
    """
    FastLanguageModel.for_inference(model)
    text_streamer = TextStreamer(tokenizer, skip_prompt=True)
    pred = zero_shot_gen(
        data=data_test,
        model=model,
        tokenizer=tokenizer,
        labels=labels,
        text_streamer=text_streamer, 
        few_shot=few_shot,
    )
    names_dataset = data_test['datasets']
    true_labels = data_test['answer']
    tmp_pred = [i if i != [] else ['Failed'] for i in pred]
    pred_flat = list(itertools.chain.from_iterable(tmp_pred))
    d_res = {'names': names_dataset, 'pred': pred_flat, 'lbl': true_labels}
    df_res = pd.DataFrame(data=d_res)
    if result_file is not None:
        df_res.to_csv(result_file, index=False)
    return df_res

def save_model(directory, model, tokenizer, quantization:str) -> None:
    """Save the model locally
    """
    model.save_pretrained_gguf(
        directory,
        tokenizer,
        quantization_method=quantization
    )