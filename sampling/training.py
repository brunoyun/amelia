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

def format_output(answer: list, labels: set) -> list:
    s = '<[|]ANSWER[|]>'
    tmp = re.split(s, answer[0])
    pred = [i for i in tmp if i in labels]
    return pred

def zero_shot_gen(
    data:Dataset,
    model,
    tokenizer,
    labels:set,
    text_streamer: TextStreamer
) -> list:
    res= []
    prompt = data['prompt']
    for prt in prompt:
        out = gen(prt, model, tokenizer, text_streamer)
        decoded_out = tokenizer.batch_decode(out)
        pred = format_output(decoded_out, labels)
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
        instruction_part="<|start_header_id|>user<|end_header_id|>\n\n",
        response_part="<|start_header_id|>assistant<|end_header_id|>\n\n"
    )
    unsloth_train(trainer)
    
def test(model, tokenizer, data_test, labels, result_file) -> pd.DataFrame:
    FastLanguageModel.for_inference(model)
    text_streamer = TextStreamer(tokenizer)
    pred = zero_shot_gen(
        data=data_test,
        model=model,
        tokenizer=tokenizer,
        labels=labels,
        text_streamer=text_streamer
    )
    names_dataset = data_test['datasets']
    true_labels = data_test['answer']
    tmp_pred = [i if i != [] else ['Failed'] for i in pred]
    pred_flat = list(itertools.chain.from_iterable(tmp_pred))
    d_res = {'names': names_dataset, 'pred': pred_flat, 'lbl': true_labels}
    df_res = pd.DataFrame(data=d_res)
    df_res.to_csv(result_file, index=False)
    return df_res