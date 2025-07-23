from unsloth import FastLanguageModel
from ollama import chat
from datasets import Dataset

import re
import itertools
import datetime
import pandas as pd
import numpy as np

import src.utils as utils
import src.training as tr
import src.prompting as prt
import src.metrics as metrics
import src.plot as plot

import src.aduc as aduc
import src.claim_detect as cd
import src.evidence_detect as ed
import src.evidence_type as et
import src.fallacies as fd
import src.quality as aq
import src.relation as arc
import src.stance_detect as sd

def get_checkpoint_path(task_name:str):
    """Get the training checkpoint folder path
    """
    checkpoint = {
        'aduc': f'./outputs/aduc/2025-05-07_17_29_07_meta-Llama-3.1-8B-Instruct_2e4000spl2_train_resp/checkpoint-250',
        'claim_detection': f'./outputs/claim_detection/2025-05-09_14_57_11_meta-Llama-3.1-8B-Instruct_2e4000spl2_train_resp/checkpoint-250',
        'evidence_detection': f'./outputs/evidence_detection/2025-05-09_15_51_24_meta-Llama-3.1-8B-Instruct_2e4000spl2_train_resp/checkpoint-250',
        'evidence_type': f'./outputs/evidence_type/2025-05-12_10_47_06_meta-Llama-3.1-8B-Instruct_2e4000spl2_train_resp/checkpoint-250',
        'fallacies': f'./outputs/fallacies/2025-05-09_16_41_44_meta-Llama-3.1-8B-Instruct_2e4000spl2_train_resp/checkpoint-340',
        'relation': f'./outputs/relation/2025-05-09_18_41_38_meta-Llama-3.1-8B-Instruct_2e4000spl2_train_resp/checkpoint-250',
        'stance_detection': f'./outputs/stance_detection/2025-05-15_09_50_35_Meta-Llama-3.1-8B-Instruct_2e4000spl2_train_resp/checkpoint-250',
        'quality': f'./outputs/quality/2025-05-12_09_44_16_meta-Llama-3.1-8B-Instruct_2e4000spl2_train_resp/checkpoint-250',
    }
    return checkpoint.get(task_name)

def get_ft_model(task_name:str):
    """Get the fine-tuned model for a specific task among [aduc, claim_detection, evidence_detection, evidence_type, fallacies, relation, stance_detection, mt_ft]
    """
    st_model = {
        'aduc': f'brunoyun/Llama-3.1-Amelia-ADUC-8B-v1', # f'./gguf_model/aduc/Llama-3.1-Amelia-ADUC-8B-v1',
        'claim_detection': f'brunoyun/Llama-3.1-Amelia-CD-8B-v1',# f'./gguf_model/claim_detection/Llama-3.1-Amelia-CD-8B-v1',
        'evidence_detection': f'brunoyun/Llama-3.1-Amelia-ED-8B-v1', # f'./gguf_model/evidence_detection/Llama-3.1-Amelia-ED-8B-v1',
        'evidence_type': f'brunoyun/Llama-3.1-Amelia-ET-8B-v1',# f'./gguf_model/evidence_type/Llama-3.1-Amelia-ET-8B-v1',
        'fallacies': f'brunoyun/Llama-3.1-Amelia-FD-8B-v1', # f'./gguf_model/fallacies/Llama-3.1-Amelia-FD-8B-v1',
        'relation': f'brunoyun/Llama-3.1-Amelia-AR-8B-v1', # f'./gguf_model/relation/Llama-3.1-Amelia-AR-8B-v1',
        'stance_detection': f'brunoyun/Llama-3.1-Amelia-SD-8B-v1', # f'./gguf_model/stance_detection/Llama-3.1-Amelia-SD-8B-v1',
        'quality': f'brunoyun/Llama-3.1-Amelia-AQA-8B-v1', # f'./gguf_model/quality/Llama-3.1-Amelia-AQA-8B-v1',
        'mt_ft': f'brunoyun/Llama-3.1-Amelia-MTFT-8B-v1', # f'./gguf_model/mt_ft/Llama-3.1-Amelia-MTFT-8B-v1',
    }
    return st_model.get(task_name)

def get_merged_model():
    """Get the merged model
    """
    return f'./merged_model/conf_della_4'

def get_syst_prompt(task_name:str):
    """Get the system prompt for a specific task
    """
    d_sys_prt = {
        'aduc': f'You are an expert in argumentation. Your task is to determine whether the given [SENTENCE] is a Claim or a Premise. Utilize the [TOPIC] and the [FULL TEXT] as context to support your decision\nYour answer must be in the following format with only Claim or Premise in the answer section:\n<|ANSWER|><answer><|ANSWER|>.',
        'claim_detection': f'You are an expert in argumentation. Your task is to determiner whether the given [SENTENCE] is a Claim or Non-claim. Utilize the [TOPIC] and the [FULL TEXT] as context to support your decision\nYour answer must be in the following format with only Claim or Non-claim in the answer section:\n<|ANSWER|><answer><|ANSWER|>.',
        'evidence_detection': f'You are an expert in argumentation. Your task is to determine whether the given [SENTENCE] is an Evidence or Non-evidence. Utilize the [TOPIC] and the [ARGUMENT] as context to support your decision\nYour answer must be in the following format with only Evidence or Non-evidence in the answer section:\n<|ANSWER|><answer><|ANSWER|>.',
        'evidence_type': f'You are an expert in argumentation. Your task is to determine the type of evidence of the given [SENTENCE]. The type of evidence would be in the [TYPE] set. Utilize the [TOPIC] and the [CLAIM] as context to support your decision\nYour answer must be in the following format with only the type of evidence in the answer section:\n<|ANSWER|><answer><|ANSWER|>.',
        'fallacies': f'You are an expert in argumentation. Your task is to determine the type of fallacy in the given [SENTENCE]. The fallacy would be in the [FALLACY] Set. Utilize the [TITLE] and the [FULL TEXT] as context to support your decision.\nYour answer must be in the following format with only the fallacy in the answer section:\n<|ANSWER|><answer><|ANSWER|>.',
        'relation': f'You are an expert in argumentation. Your task is to determine the type of relation between [SOURCE] and [TARGET]. The type of relation would be in the [RELATION] set. Utilize the [TOPIC] as context to support your decision\nYour answer must be in the following format with only the type of the relation in the answer section:\n<|ANSWER|><answer><|ANSWER|>.',
        'stance_detection': f'You are an expert in argumentation. Your task is to determine whether the given [SENTENCE] is For or Against. Utilize the [TOPIC] as context to support your decision\nYour answer must be in the following format with only For or Against in the answer section:\n<|ANSWER|><answer><|ANSWER|>.',
        'quality': f'You are an expert in argumentation. Your task is to determine the quality of the [SENTENCE]. The quality would be in the [QUALITY] set. Utilize the [TOPIC], the [STANCE] and the [DEFINITION] as context to support your decision\nYour answer must be in the following format with only the quality in the answer section:\n<|ANSWER|><answer><|ANSWER|>.'
    }
    return d_sys_prt.get(task_name)

def get_examples_aduc():
    """Get the few-shot examples for the ACC task
    """
    ex = [
            {'role': 'user', 'content': "[TOPIC]: allow_shops_to_open_on_holidays_and_sundays\n[SENTENCE]: It is quite clear that Sunday should remain a day of rest.\n[FULL TEXT]: It is quite clear that Sunday should remain a day of rest.For you can go shopping 6 days a week,as shops are open till 10pm on some days.Thus Sundays should remain a day for getting together with family,otherwise there will be a new kind of leisure activity where people spend their time in supermarkets instead of outside or in conversation etc.Furthermore, there's no need to keep shops open on Sundays.\n"},
            {'role': 'assistant', 'content': "<|ANSWER|>Claim<|ANSWER|>."},
            {'role': 'user', 'content': "[TOPIC]: introduce_capital_punishment\n[SENTENCE]: Even if many people think that a murderer has already decided on the life or death of another person,\n[FULL TEXT]: The death penalty is a legal means that as such is not practicable in Germany.For one thing, inviolable human dignity is anchored in our constitution,and furthermore no one may have the right to adjudicate upon the death of another human being.Even if many people think that a murderer has already decided on the life or death of another person,this is precisely the crime that we should not repay with the same.\n"},
            {'role': 'assistant', 'content': "<|ANSWER|>Premise<|ANSWER|>."}
    ]
    return ex

def get_examples_cd():
    """Get the few-shot examples for the CD task
    """
    ex =  [
            {'role': 'user', 'content': "[TOPIC]: Should sex education courses be compulsory in middle schools?\n[SENTENCE]: Rubin and Kindendall expressed that sex education is not merely the topics of reproduction and teaching how babies are conceived and born.\n[FULL TEXT]: Rubin and Kindendall expressed that sex education is not merely the topics of reproduction and teaching how babies are conceived and born.\n"},
            {'role': 'assistant', 'content': "<|ANSWER|>Claim<|ANSWER|>."},
            {'role': 'user', 'content': "[TOPIC]: Should we ban smoking in public places\n[SENTENCE]: A dilemma arises - removing smoke breaks is not an option either, now that smokers are used to taking smoke breaks, and since doing so could decrease their productivity even more in the short run.\n[FULL TEXT]: A dilemma arises - removing smoke breaks is not an option either, now that smokers are used to taking smoke breaks, and since doing so could decrease their productivity even more in the short run.\n"},
            {'role': 'assistant', 'content': "<|ANSWER|>Non-claim<|ANSWER|>."}
    ]
    return ex

def get_examples_ed():
    """Get the few-shot examples for ED task
    """
    ex = [
            {'role': 'user', 'content': "[TOPIC]: We should subsidize vocational education\n[ARGUMENT]: subsidizing vocational education is expensive\n[SENTENCE]: A report by the National Skills Coalition in 2019 found that subsidizing vocational education can reduce funding for other important programs, such as financial aid, student support services, and research and development.\n"},
            {'role': 'assistant', 'content': "<|ANSWER|>Evidence<|ANSWER|>."},
            {'role': 'user', 'content': "[TOPIC]: We should abolish capital punishment\n[ARGUMENT]: The death penalty helps the victim/their family\n[SENTENCE]: A November 2009 television survey showed that 70% favoured reinstating the death penalty for at least one of the following crimes: armed robbery, rape, crimes related to paedophilia, terrorism, adult murder, child murder, child rape, treason, child abuse or kidnapping.\n"},
            {'role': 'assistant', 'content': "<|ANSWER|>Non-evidence<|ANSWER|>."}
    ]
    return ex

def get_examples_et():
    """Get the few-shot exampled for the ET task
    """
    ex = [
         {'role': 'user', 'content': "[TYPE]: {'NONE', 'ANECDOTAL', 'EXPERT', 'EXPLANATION', 'STUDY'}\n[TOPIC]: Should we cancel the standardized test for university entrance\n[CLAIM]: “Standardized tests can level the playing field for low-income and rural college applicants,” writes Rich Saunders for the Chronicle of Higher Education.\n[SENTENCE]: Still, some say that standardized testing is one of the most objective measures currently at schools disposal for assessing student achievement and potential.\n"},
         {'role': 'assistant', 'content': '<|ANSWER|>NONE<|ANSWER|>.'},
         {'role': 'user', 'content': "[TYPE]: {'NONE', 'ANECDOTAL', 'EXPERT', 'EXPLANATION', 'STUDY'}\n[TOPIC]: Should we ban beauty pageants?\n[CLAIM]: They promote local economic opportunities.\n[SENTENCE]: Miss America 1999 because the first winner with diabetes and publicized the use of an insulin pump.\n"},
         {'role': 'assistant', 'content': '<|ANSWER|>ANECDOTAL<|ANSWER|>.'},
         {'role': 'user', 'content': "[TYPE]: {'NONE', 'ANECDOTAL', 'EXPERT', 'EXPLANATION', 'STUDY'}\n[TOPIC]: Should abortion be prohibited\n[CLAIM]: David C. Nice, of the University of Georgia, describes support for anti-abortion violence as a political weapon against women's rights, one that is associated with tolerance for violence toward women [ref].\n[SENTENCE]: Numerous organizations have also recognized anti-abortion extremism as a form of Christian terrorism [ref].\n"},
         {'role': 'assistant', 'content': '<|ANSWER|>EXPERT<|ANSWER|>.'},
         {'role': 'user', 'content': "[TYPE]: {'NONE', 'ANECDOTAL', 'EXPERT', 'EXPLANATION', 'STUDY'}\n[TOPIC]: Should developing countries restrict rural-to-urban migration?\n[CLAIM]: Restrictions on migration would benefit people in the cities economically and socially\n[SENTENCE]: Thus, people who enter the city cannot find work, as production does not grow in relation to the people who enter.\n"},
         {'role': 'assistant', 'content': '<|ANSWER|>EXPLANATION<|ANSWER|>.'},
         {'role': 'user', 'content': "[TYPE]: {'NONE', 'ANECDOTAL', 'EXPERT', 'EXPLANATION', 'STUDY'}\n[TOPIC]: Should we legalize same-sex marriage?\n[CLAIM]: The establishment of same-sex marriage is associated with a significant reduction in the rate of attempted suicide among children, with the effect being concentrated among children of a minority sexual orientation.\n[SENTENCE]: No reduction in the rate of attempted suicide among children occurred in a particular state until that state recognized same-sex marriage.\n"},
         {'role': 'assistant', 'content': '<|ANSWER|>STUDY<|ANSWER|>.'}
    ]
    return ex
    
def get_examples_fd():
    """Get the few-shot examples for the FD task
    """
    example = [
        {'role': 'user', 'content': "[FALLACY]: {'appeal to ridicule', 'none', 'appeal to worse problems', 'straw man', 'causal oversimplification', 'guilt by association', 'false analogy', 'ad hominem', 'appeal to authority', 'appeal to fear', 'appeal to tradition', 'slippery slope', 'circular reasoning', 'ad populum', 'hasty generalization', 'false dilemma', 'equivocation', 'appeal to nature', 'appeal to majority', 'false causality'}\n[TITLE]: unknown\n[SENTENCE]: This vice president has been an advocate for over a decade for lifting sanctions against Iran, the largest state sponsor of terrorism on the planet.\n[FULL TEXT]: Iran has moved forward with its nuclear weapons program. They're more dangerous today than they were four years ago. North Korea has moved forward with their nuclear weapons program, gone from one to two nuclear weapons to six to eight nuclear weapons. This vice president has been an advocate for over a decade for lifting sanctions against Iran, the largest state sponsor of terrorism on the planet. It's a mistake.\n\n"},
        {'role': 'assistant', 'content': "<|ANSWER|>appeal to fear<|ANSWER|>."},
        {'role': 'user', 'content': "[FALLACY]: {'appeal to ridicule', 'none', 'appeal to worse problems', 'straw man', 'causal oversimplification', 'guilt by association', 'false analogy', 'ad hominem', 'appeal to authority', 'appeal to fear', 'appeal to tradition', 'slippery slope', 'circular reasoning', 'ad populum', 'hasty generalization', 'false dilemma', 'equivocation', 'appeal to nature', 'appeal to majority', 'false causality'}\n[TITLE]: The most critical lesson we've learned from the COVID-19 outbreak is the importance of free speech\n[SENTENCE]: We now know for a fact that the responsibility of COVID lies with China, or more specifically, the CCP. How many other horrendous diseases have they been responsible for? They seem to always be at the center of these health crises. The true amount of death caused by their negligence must be monumental.\n[FULL TEXT]: We now know for a fact that the responsibility of COVID lies with China, or more specifically, the CCP. How many other horrendous diseases have they been responsible for? They seem to always be at the center of these health crises. The true amount of death caused by their negligence must be monumental.\n"},
        {'role': 'assistant', 'content': "<|ANSWER|>hasty generalization<|ANSWER|>."},
        {'role': 'user', 'content': "[FALLACY]: {'appeal to ridicule', 'none', 'appeal to worse problems', 'straw man', 'causal oversimplification', 'guilt by association', 'false analogy', 'ad hominem', 'appeal to authority', 'appeal to fear', 'appeal to tradition', 'slippery slope', 'circular reasoning', 'ad populum', 'hasty generalization', 'false dilemma', 'equivocation', 'appeal to nature', 'appeal to majority', 'false causality'}\n[TITLE]: An owl refuses to leave Tanzanian parliament. What does it all mean?\n[SENTENCE]: At least the country of Tanzania has a multi party system. If it did not exist at all, then there would not be free elections, which there are. Some people see a decline in this system with the new amendments that are proposed, but at least the multi party system is still left intact.\n[FULL TEXT]: At least the country of Tanzania has a multi party system. If it did not exist at all, then there would not be free elections, which there are. Some people see a decline in this system with the new amendments that are proposed, but at least the multi party system is still left intact.\n"},
        {'role': 'assistant', 'content': "<|ANSWER|>appeal to worse problems<|ANSWER|>."},
        {'role': 'user', 'content': "[FALLACY]: {'appeal to ridicule', 'none', 'appeal to worse problems', 'straw man', 'causal oversimplification', 'guilt by association', 'false analogy', 'ad hominem', 'appeal to authority', 'appeal to fear', 'appeal to tradition', 'slippery slope', 'circular reasoning', 'ad populum', 'hasty generalization', 'false dilemma', 'equivocation', 'appeal to nature', 'appeal to majority', 'false causality'}\n[TITLE]: Twitter reveals China's information operations on Hong Kong protests\n[SENTENCE]: Isn't China the rightful authority of Hong Kong? Why should HK have any more sovereignty than LA or NYC or something? All the experts I've ever heard from make clear that Hong Kong is an integral part of the PRC and so I don't see why China's authority should be questioned here.\n[FULL TEXT]: We have to speak up against bullying of Hong Kong by China. If we do not, Hong Kong will lose its sovereignty. This will lead to more global control by China. Isn't China the rightful authority of Hong Kong? Why should HK have any more sovereignty than LA or NYC or something? All the experts I've ever heard from make clear that Hong Kong is an integral part of the PRC and so I don't see why China's authority should be questioned here.\n"},
        {'role': 'assistant', 'content': "<|ANSWER|>appeal to authority<|ANSWER|>."},
        {'role': 'user', 'content': "[FALLACY]: {'appeal to ridicule', 'none', 'appeal to worse problems', 'straw man', 'causal oversimplification', 'guilt by association', 'false analogy', 'ad hominem', 'appeal to authority', 'appeal to fear', 'appeal to tradition', 'slippery slope', 'circular reasoning', 'ad populum', 'hasty generalization', 'false dilemma', 'equivocation', 'appeal to nature', 'appeal to majority', 'false causality'}\n[TITLE]: unknown\n[SENTENCE]: Mr. Governor issues a proclamation for the people of his state to pray for rain.\n[FULL TEXT]: Mr. Governor issues a proclamation for the people of his state to pray for rain. Several months later, it rains. Praise the gods!\n\n"},
        {'role': 'assistant', 'content': "<|ANSWER|>false causality<|ANSWER|>."},
        {'role': 'user', 'content': "[FALLACY]: {'appeal to ridicule', 'none', 'appeal to worse problems', 'straw man', 'causal oversimplification', 'guilt by association', 'false analogy', 'ad hominem', 'appeal to authority', 'appeal to fear', 'appeal to tradition', 'slippery slope', 'circular reasoning', 'ad populum', 'hasty generalization', 'false dilemma', 'equivocation', 'appeal to nature', 'appeal to majority', 'false causality'}\n[TITLE]: Documentary exposes the threat of facial recognition surveillance in Serbia\n[SENTENCE]: I'm not an advocate of citizens surveillance. I would prefer to go back to the old days when we just had cameras to watch for crimes being committed. That was the traditional use. Now they become invasive. Most people see this as a loss of privacy. The cameras used to just record anything that passed, now they follow you. You can record the store without recording the shoppers.\n[FULL TEXT]: I'm not an advocate of citizens surveillance. I would prefer to go back to the old days when we just had cameras to watch for crimes being committed. That was the traditional use. Now they become invasive. Most people see this as a loss of privacy. The cameras used to just record anything that passed, now they follow you. You can record the store without recording the shoppers.\n"},
        {'role': 'assistant', 'content': "<|ANSWER|>appeal to tradition<|ANSWER|>."},
        {'role': 'user', 'content': "[FALLACY]: {'appeal to ridicule', 'none', 'appeal to worse problems', 'straw man', 'causal oversimplification', 'guilt by association', 'false analogy', 'ad hominem', 'appeal to authority', 'appeal to fear', 'appeal to tradition', 'slippery slope', 'circular reasoning', 'ad populum', 'hasty generalization', 'false dilemma', 'equivocation', 'appeal to nature', 'appeal to majority', 'false causality'}\n[TITLE]: unknown\n[SENTENCE]: The Democratic party clearly has not learned the right lesson from Hillary Clinton's miserable failure.\n[FULL TEXT]: TITLE: Discussion Thread (Part 3): 2020 Presidential Race Democratic Debates - Post Debate | Night 2 POST: Joe Biden will lose to Trump if he is the nominee. The Democratic party clearly has not learned the right lesson from Hillary Clinton's miserable failure. NOBODY WANTS ESTABLISHMENT POLITICIANS ANYMORE. NOBODY LIKES THE STATUS QUO. Like Jesus Christ you think they would learn. POST: The status quo in America is that its the best its ever been. We live in one of the best societies in the best times that humans have ever experienced.\n\n"},
        {'role': 'assistant', 'content': "<|ANSWER|>ad populum<|ANSWER|>."},
        {'role': 'user', 'content': "[FALLACY]: {'appeal to ridicule', 'none', 'appeal to worse problems', 'straw man', 'causal oversimplification', 'guilt by association', 'false analogy', 'ad hominem', 'appeal to authority', 'appeal to fear', 'appeal to tradition', 'slippery slope', 'circular reasoning', 'ad populum', 'hasty generalization', 'false dilemma', 'equivocation', 'appeal to nature', 'appeal to majority', 'false causality'}\n[TITLE]: unknown\n[SENTENCE]: The Democratic party clearly has not learned the right lesson from Hillary Clinton's miserable failure.\n[FULL TEXT]: TITLE: Discussion Thread (Part 3): 2020 Presidential Race Democratic Debates - Post Debate | Night 2 POST: Joe Biden will lose to Trump if he is the nominee. The Democratic party clearly has not learned the right lesson from Hillary Clinton's miserable failure. NOBODY WANTS ESTABLISHMENT POLITICIANS ANYMORE. NOBODY LIKES THE STATUS QUO. Like Jesus Christ you think they would learn. POST: The status quo in America is that its the best its ever been. We live in one of the best societies in the best times that humans have ever experienced.\n\n"},
        {'role': 'assistant', 'content': "<|ANSWER|>guilt by association<|ANSWER|>."},
        {'role': 'user', 'content': "[FALLACY]: {'appeal to ridicule', 'none', 'appeal to worse problems', 'straw man', 'causal oversimplification', 'guilt by association', 'false analogy', 'ad hominem', 'appeal to authority', 'appeal to fear', 'appeal to tradition', 'slippery slope', 'circular reasoning', 'ad populum', 'hasty generalization', 'false dilemma', 'equivocation', 'appeal to nature', 'appeal to majority', 'false causality'}\n[TITLE]: unknown\n[SENTENCE]:  So don't you see that it's silly to continue believing in God?\n[FULL TEXT]: According to Freud, your belief in God stems from your need for a strong father figure. So don't you see that it's silly to continue believing in God?\n\n"},
        {'role': 'assistant', 'content': '<|ANSWER|>causal oversimplification<|ANSWER|>.'},
        {'role': 'user', 'content': "[FALLACY]: {'appeal to ridicule', 'none', 'appeal to worse problems', 'straw man', 'causal oversimplification', 'guilt by association', 'false analogy', 'ad hominem', 'appeal to authority', 'appeal to fear', 'appeal to tradition', 'slippery slope', 'circular reasoning', 'ad populum', 'hasty generalization', 'false dilemma', 'equivocation', 'appeal to nature', 'appeal to majority', 'false causality'}\n[TITLE]: Watchdog decries threat to press freedom as China expels some US journalists\n[SENTENCE]: We're going to have to get ready to fight with China. It's either that or allow them to walk all over us. Perhaps some politician can think of something else to do that might get them to stick to their purposed Freedom of the Press.\n[FULL TEXT]: We're going to have to get ready to fight with China. It's either that or allow them to walk all over us. Perhaps some politician can think of something else to do that might get them to stick to their purposed Freedom of the Press.\n"},
        {'role': 'assistant', 'content': "<|ANSWER|>false dilemma<|ANSWER|>."},
        {'role': 'user', 'content': "[FALLACY]: {'appeal to ridicule', 'none', 'appeal to worse problems', 'straw man', 'causal oversimplification', 'guilt by association', 'false analogy', 'ad hominem', 'appeal to authority', 'appeal to fear', 'appeal to tradition', 'slippery slope', 'circular reasoning', 'ad populum', 'hasty generalization', 'false dilemma', 'equivocation', 'appeal to nature', 'appeal to majority', 'false causality'}\n[TITLE]: unknown\n[SENTENCE]: Your life is fucked because someone decided your life should be shit for being a horny teenager.\n[FULL TEXT]: TITLE: The sex offender registry should only be for people for individuals who pose a risk to the safety of others. Not some 16 year old who sent a dick pic!\r\nPOST: Imagine being a horny naive 16 year old. You were talking to a girl, you guys exchange nudes. You're 21. It's been revealed to other adults that you sent a picture of your dick (nice cock btw). You're on the sex offender registry. You will get almost all job offers turned down. Your life is fucked because someone decided your life should be shit for being a horny teenager.\r\nPOST: 21 year old aren't teenagers\n"},
        {'role': 'assistant', 'content': "<|ANSWER|>appeal to ridicule<|ANSWER|>."},
        {'role': 'user', 'content': "[FALLACY]: {'appeal to ridicule', 'none', 'appeal to worse problems', 'straw man', 'causal oversimplification', 'guilt by association', 'false analogy', 'ad hominem', 'appeal to authority', 'appeal to fear', 'appeal to tradition', 'slippery slope', 'circular reasoning', 'ad populum', 'hasty generalization', 'false dilemma', 'equivocation', 'appeal to nature', 'appeal to majority', 'false causality'}\n[TITLE]: unknown\n[SENTENCE]: Carbon dioxide hurts nobody' s health.\n[FULL TEXT]: Carbon dioxide hurts nobody' s health. It' s good for plants. Climate change need not endanger anyone.\n\n"},
        {'role': 'assistant', 'content': "<|ANSWER|>false analogy<|ANSWER|>."},
        {'role': 'user', 'content': "[FALLACY]: {'appeal to ridicule', 'none', 'appeal to worse problems', 'straw man', 'causal oversimplification', 'guilt by association', 'false analogy', 'ad hominem', 'appeal to authority', 'appeal to fear', 'appeal to tradition', 'slippery slope', 'circular reasoning', 'ad populum', 'hasty generalization', 'false dilemma', 'equivocation', 'appeal to nature', 'appeal to majority', 'false causality'}\n[TITLE]: unknown\n[SENTENCE]:  If indoor smoking laws are passed for bars, the bars will go out of business since people who drink, smoke while they drink.\n[FULL TEXT]: If indoor smoking laws are passed for bars, the bars will go out of business since people who drink, smoke while they drink.\n\n"},
        {'role': 'assistant', 'content': '<|ANSWER|>slippery slope<|ANSWER|>.'},
        {'role': 'user', 'content': "[FALLACY]: {'appeal to ridicule', 'none', 'appeal to worse problems', 'straw man', 'causal oversimplification', 'guilt by association', 'false analogy', 'ad hominem', 'appeal to authority', 'appeal to fear', 'appeal to tradition', 'slippery slope', 'circular reasoning', 'ad populum', 'hasty generalization', 'false dilemma', 'equivocation', 'appeal to nature', 'appeal to majority', 'false causality'}\n[TITLE]: Trinidad & Tobago deports Venezuelan women and children as matter of ‘national security’\n[SENTENCE]: I'd really be interested in seeing opinion polls about what most people think about this issue. I'm not so interested in what some handpicked Twitter activists or NGO figures are saying. Whatever the majority decides should be what dictates the country's policy, nothing else.\n[FULL TEXT]: I'd really be interested in seeing opinion polls about what most people think about this issue. I'm not so interested in what some handpicked Twitter activists or NGO figures are saying. Whatever the majority decides should be what dictates the country's policy, nothing else.\n"},
        {'role': 'assistant', 'content': "<|ANSWER|>appeal to majority<|ANSWER|>."},
        {'role': 'user', 'content': "[FALLACY]: {'appeal to ridicule', 'none', 'appeal to worse problems', 'straw man', 'causal oversimplification', 'guilt by association', 'false analogy', 'ad hominem', 'appeal to authority', 'appeal to fear', 'appeal to tradition', 'slippery slope', 'circular reasoning', 'ad populum', 'hasty generalization', 'false dilemma', 'equivocation', 'appeal to nature', 'appeal to majority', 'false causality'}\n[TITLE]: Singapore Authorities Ban Documentary on Palestinian Teen Activists for ‘Skewed Narrative’\n[SENTENCE]: It is common to want to make sure that friendly relations are not strained by something as simple as a documentary. I think that is what is happening here. Singapore does not want to upset the Israelis over a documentary that is basically framed as propaganda. A country is right to want to protect its best interests first, and that is a natural reaction. I understand the want for freedom of expression and I also think that the movie should be shown, but I understand why the government would be tense about it, but that does not mean that freedoms of expression should be stifled. Perhaps there could be a disclaimer or some kind of statement from the Singapore film festival and/or the government stating that this film was the view of the filmmaker and does not necessarily fall in line with broader views maybe? This way the important factor of allowing of free speech is met and the government can distance itself from angering an ally.\n[FULL TEXT]: It is common to want to make sure that friendly relations are not strained by something as simple as a documentary. I think that is what is happening here. Singapore does not want to upset the Israelis over a documentary that is basically framed as propaganda. A country is right to want to protect its best interests first, and that is a natural reaction. I understand the want for freedom of expression and I also think that the movie should be shown, but I understand why the government would be tense about it, but that does not mean that freedoms of expression should be stifled. Perhaps there could be a disclaimer or some kind of statement from the Singapore film festival and/or the government stating that this film was the view of the filmmaker and does not necessarily fall in line with broader views maybe? This way the important factor of allowing of free speech is met and the government can distance itself from angering an ally.\n"},
        {'role': 'assistant', 'content': "<|ANSWER|>appeal to nature<|ANSWER|>."},
        {'role': 'user', 'content': "[FALLACY]: {'appeal to ridicule', 'none', 'appeal to worse problems', 'straw man', 'causal oversimplification', 'guilt by association', 'false analogy', 'ad hominem', 'appeal to authority', 'appeal to fear', 'appeal to tradition', 'slippery slope', 'circular reasoning', 'ad populum', 'hasty generalization', 'false dilemma', 'equivocation', 'appeal to nature', 'appeal to majority', 'false causality'}\n[TITLE]: ‘Luanda Leaks': How Africa's richest woman plundered the Angolan state\n[SENTENCE]: This is a prime example of the type of corruption that comes from relatives in government. Those that are related should not be allowed to profit off each other in public positions.\n[FULL TEXT]: This is a prime example of the type of corruption that comes from relatives in government. Those that are related should not be allowed to profit off each other in public positions.\n"},
        {'role': 'assistant', 'content': "<|ANSWER|>none<|ANSWER|>."},
        {'role': 'user', 'content': "[FALLACY]: {'appeal to ridicule', 'none', 'appeal to worse problems', 'straw man', 'causal oversimplification', 'guilt by association', 'false analogy', 'ad hominem', 'appeal to authority', 'appeal to fear', 'appeal to tradition', 'slippery slope', 'circular reasoning', 'ad populum', 'hasty generalization', 'false dilemma', 'equivocation', 'appeal to nature', 'appeal to majority', 'false causality'}\n[TITLE]: unknown\n[SENTENCE]: Dont sit here and try and talk about realism or how you shouldnt be able to swap instantly between seats when there is hardly anything realistic about BF vehicle play in general.\n[FULL TEXT]: TITLE: What would people think about preventing people from switching out from a pilot seat? i.e. to a gunner position; you'd still be able to bail. POST: This is a ridiculous post and this mechanic will never be changed. Seat switching has always been a BF mechanic. Dont sit here and try and talk about realism or how you shouldnt be able to swap instantly between seats when there is hardly anything realistic about BF vehicle play in general. Sounds like you need to get better if you are having trouble shooting down the most vulnerable aircraft in the sky.\n\n"},
        {'role': 'assistant', 'content': "<|ANSWER|>straw man<|ANSWER|>."},
        {'role': 'user', 'content': "[FALLACY]: {'appeal to ridicule', 'none', 'appeal to worse problems', 'straw man', 'causal oversimplification', 'guilt by association', 'false analogy', 'ad hominem', 'appeal to authority', 'appeal to fear', 'appeal to tradition', 'slippery slope', 'circular reasoning', 'ad populum', 'hasty generalization', 'false dilemma', 'equivocation', 'appeal to nature', 'appeal to majority', 'false causality'}\n[TITLE]: unknown\n[SENTENCE]: We know God exists because he made everything\n[FULL TEXT]: We know God exists because he made everything\n\n"},
        {'role': 'assistant', 'content': "<|ANSWER|>circular reasoning<|ANSWER|>."},
        {'role': 'user', 'content': '[FALLACY]: {\'appeal to ridicule\', \'none\', \'appeal to worse problems\', \'straw man\', \'causal oversimplification\', \'guilt by association\', \'false analogy\', \'ad hominem\', \'appeal to authority\', \'appeal to fear\', \'appeal to tradition\', \'slippery slope\', \'circular reasoning\', \'ad populum\', \'hasty generalization\', \'false dilemma\', \'equivocation\', \'appeal to nature\', \'appeal to majority\', \'false causality\'}\n[TITLE]: unknown\n[SENTENCE]: So the least corrupt state is anarchy by this logic.\n[FULL TEXT]: TITLE: The more corrupt the state, the more numerous the laws. - Tacitus[529x529] POST: So the least corrupt state is anarchy by this logic. This is""deep"" libertarian bullshit. Yes, over regulation is a thing, but laws against theft, murder, slavery, child labor, education, etc. are good things - not corruption. POST: > laws against theft, murder, slavery. Good people don\'t need laws to tell them how to be good, bad people either ignore laws or use/create them to hide behind...\n\n'},
        {'role': 'assistant', 'content': "<|ANSWER|>equivocation<|ANSWER|>."},
        {'role': 'user', 'content': "[FALLACY]: {'appeal to ridicule', 'none', 'appeal to worse problems', 'straw man', 'causal oversimplification', 'guilt by association', 'false analogy', 'ad hominem', 'appeal to authority', 'appeal to fear', 'appeal to tradition', 'slippery slope', 'circular reasoning', 'ad populum', 'hasty generalization', 'false dilemma', 'equivocation', 'appeal to nature', 'appeal to majority', 'false causality'}\n[TITLE]: unknown\n[SENTENCE]: Why not a brain surgeon?\n[FULL TEXT]: TITLE: Can I get into finance with a Law degree? POST: I have a JD, an MBA, and an MSF so I am pretty well versed in the skill sets. What about a JD would make you think you could do finance? It is a pretty in depth math program. There are some advanced math skills needed that are covered nowhere in undergrad or a JD unless you were also a finance major for BS. This is law school arrogance at its finest. Why not a brain surgeon?\n\n"},
        {'role': 'assistant', 'content': "<|ANSWER|>ad hominem<|ANSWER|>."},
    ]
    return example

def get_examples_ar():
    """Get the few-shot examples for the AR task
    """
    ex = [
            {'role': 'user', 'content': "[RELATION]: {'support', 'no relation', 'attack'}\n[TOPIC]: higher_dog_poo_fines\n[SOURCE]: due to the dirt, the stench and the often considerable effort to get rid of it.\n[TARGET]: Stepping in dog dirt is gross and absolutely ruins your day\n"},
            {'role': 'assistant', 'content': "<|ANSWER|>support<|ANSWER|>."},
            {'role': 'user', 'content': "[RELATION]: {'support', 'no relation', 'attack'}\n[TOPIC]: allow_shops_to_open_on_holidays_and_sundays\n[SOURCE]: so this has hardly any effect on the level of full-time employment and the number of unemployed.\n[TARGET]: Due to the increase in opening hours on Sundays and holidays there is a rise in employment. For these reasons supermarkets and shopping centres should not be allowed to be open for business on arbitrary Sundays and holidays.\n"},
            {'role': 'assistant', 'content': "<|ANSWER|>attack<|ANSWER|>."},
            {'role': 'user', 'content': "[RELATION]: {'support', 'no relation', 'attack'}\n[TOPIC]: disarmament\n[SOURCE]: [ Applause . ] Secondly , I think if we can achieve a level of parity with the Communists , then we will be able to talk about disarmament . Winston Churchill said 10 years ago , We arm to parley .\n[TARGET]: We have shown very little interest in their health , welfare , and economic problems . Third , I think the United States should put greater emphasis on disarmament . Fourth , I think it has been the greatest blow we have had in the 1950 's or since World War II , when the United States was second in space .\n"},
            {'role': 'assistant', 'content': "<|ANSWER|>no relation<|ANSWER|>."},
    ]
    return ex

def get_examples_sd():
    """Get the few-shot examples for the SD task
    """
    ex = [
            {'role': 'user', 'content': "[TOPIC]: This house would ban gambling\n[SENTENCE]: problem gambling has been shown to cause dysfunctional families\n"},
            {'role': 'assistant', 'content': "<|ANSWER|>For<|ANSWER|>."},
            {'role': 'user', 'content': "[TOPIC]: This house believes that Israel\'s 2008-2009 military operations against Gaza were justified\n[SENTENCE]: Israel\'s military assault on Gaza was designed to ""humiliate and terrorize a civilian population\n"},
            {'role': 'assistant', 'content': "<|ANSWER|>Against<|ANSWER|>."},
    ]
    return ex

def get_examples_quality(quality_dim:str):
    """Get the few-shot examples for the AQ task
    """
    examples = {
        'overall_quality': [
            {'role': 'user', 'content': "[QUALITY]: {'Average', 'High', 'Low'}\n[TOPIC]: personal-pursuit-or-advancing-the-common-good\n[STANCE]: personal-pursuit\n[DEFINITION]: overall_quality: Argumentation quality in total\n[SENTENCE]: Human nature is to endeavor personal pursuit. If you assume that human nature is to endeavor advancing the common good, then you're going to get screwed by someone who endeavors personal pursuit. Most Americans endeavor advancing the common good; which is why they get screwed by politicians who endeavor personal pursuit. If everyone endeavors personal pursuit, then we would keep each other in check. It is easier to get most people to endeavor personal pursuit than it is to get most people to endeavor advancing the common good; which is why communism failed.\n"},
            {'role': 'assistant', 'content': "<|ANSWER|>Low<|ANSWER|>.\n"},
            {'role': 'user', 'content': "[QUALITY]: {'Average', 'High', 'Low'}\n[TOPIC]: india-has-the-potential-to-lead-the-world\n[STANCE]: yes-for\n[DEFINITION]: overall_quality:  Argumentation quality in total\n[SENTENCE]: I believe that India is fast emerging as a hub for international trade and investment. India has provided a huge opportunity to enhance trade and investments in sectors such as mechanical and electrical engineering, food processing, automotive, tourism and banking among others. Almost every multinational company is focusing on India.\n"},
            {'role': 'assistant', 'content': "<|ANSWER|>Average<|ANSWER|>."},
            {'role': 'user', 'content': "[QUALITY]: {'Average', 'High', 'Low'}\n[TOPIC]: if-your-spouse-committed-murder-and-he-or-she-confided-in-you-would-you-turn-them-in\n[STANCE]: yes\n[DEFINITION]: overall_quality:  Argumentation quality in total\n[SENTENCE]: Murder under any circumstance is not right. A person who commits the act knows the consequences, and they know that they are guilty. I wouldn't be able to look at my spouse the same ever again. I would not be able to live with the secret of a murder. <br/> So yea, i would turn them in.\n"},
            {'role': 'assistant', 'content': "<|ANSWER|>High<|ANSWER|>."}
        ],
        'local_acceptability': [
            {'role': 'user', 'content': "[QUALITY]: {'Average', 'High', 'Low'}\n[TOPIC]: personal-pursuit-or-advancing-the-common-good\n[STANCE]: personal-pursuit\n[DEFINITION]: local_acceptability: A premise of an argument is acceptable if it is rationally worthy of being believed to be true\n[SENTENCE]: Human nature is to endeavor personal pursuit. If you assume that human nature is to endeavor advancing the common good, then you're going to get screwed by someone who endeavors personal pursuit. Most Americans endeavor advancing the common good; which is why they get screwed by politicians who endeavor personal pursuit. If everyone endeavors personal pursuit, then we would keep each other in check. It is easier to get most people to endeavor personal pursuit than it is to get most people to endeavor advancing the common good; which is why communism failed.\n"},
            {'role': 'assistant', 'content': "<|ANSWER|>Low<|ANSWER|>."},
            {'role': 'user', 'content': "[QUALITY]: {'Average', 'High', 'Low'}\n[TOPIC]: india-has-the-potential-to-lead-the-world\n[STANCE]: yes-for\n[DEFINITION]: local_acceptability: A premise of an argument is acceptable if it is rationally worthy of being believed to be true\n[SENTENCE]: I believe that India is fast emerging as a hub for international trade and investment. India has provided a huge opportunity to enhance trade and investments in sectors such as mechanical and electrical engineering, food processing, automotive, tourism and banking among others. Almost every multinational company is focusing on India.\n"},
            {'role': 'assistant', 'content': "<|ANSWER|>Average<|ANSWER|>."},
            {'role': 'user', 'content': "[QUALITY]: {'Average', 'High', 'Low'}\n[TOPIC]: if-your-spouse-committed-murder-and-he-or-she-confided-in-you-would-you-turn-them-in\n[STANCE]: yes\n[DEFINITION]: local_acceptability: A premise of an argument is acceptable if it is rationally worthy of being believed to be true\n[SENTENCE]: Murder under any circumstance is not right. A person who commits the act knows the consequences, and they know that they are guilty. I wouldn't be able to look at my spouse the same ever again. I would not be able to live with the secret of a murder. <br/> So yea, i would turn them in.\n"},
            {'role': 'assistant', 'content': "<|ANSWER|>High<|ANSWER|>."},
        ],
        'appropriateness': [
            {'role': 'user', 'content': "[QUALITY]: {'Average', 'High', 'Low'}\n[TOPIC]: should-physical-education-be-mandatory-in-schools\n[STANCE]: no\n[DEFINITION]: appropriateness: Argumentation has an appropriate style if the used language supports the creation of credibility and emotions as well as if it is proportional to the issue\n[SENTENCE]: Sport should not be compulsaryy!!!!!!!!!!!!!!! <br/> So man kids get bullied in pe because they are fatter then others or more unfit! Bullies pick there victims each day in pe so all u guys on the affirmative team are WRONG!\n"},
            {'role': 'assistant', 'content': "<|ANSWER|>Low<|ANSWER|>."},
            {'role': 'user', 'content': "[QUALITY]: {'Average', 'High', 'Low'}\n[TOPIC]: human-growth-and-development-should-parents-use-spanking-as-an-option-to-discipline\n[STANCE]: no\n[DEFINITION]: appropriateness: Argumentation has an appropriate style if the used language supports the creation of credibility and emotions as well as if it is proportional to the issue\n[SENTENCE]: A parent shouldn't use spanking as an option to discipline a child because they might grow up traumatized and they might feel unloved since they would get spanked a lot, that will cause a low self of esteem. Also spanking shouldn't be an option because at that moment the parent would be full with anger and they wouldn't be thinking logically.\n"},
            {'role': 'assistant', 'content': "<|ANSWER|>Average<|ANSWER|>."},
            {'role': 'user', 'content': "[QUALITY]: {'Average', 'High', 'Low'}\n[TOPIC]: william-farquhar-ought-to-be-honoured-as-the-rightful-founder-of-singapore\n[STANCE]: yes-of-course\n[DEFINITION]: appropriateness: Argumentation has an appropriate style if the used language supports the creation of credibility and emotions as well as if it is proportional to the issue\n[SENTENCE]: To me, a founder would be a leader, well respected by the organisation/group he is managing. Both Raffles and Farquhar were good leaders. However, when William Farquhar was sent off to Britain, more people went to send him off than the combined number of people that sent Raffles off during the three times Raffles left Singapore. This shows that the people of Singapore respected Farquhar more than Raffles, and to them, Farquhar was a more competent leader.\n"},
            {'role': 'assistant', 'content': "<|ANSWER|>High<|ANSWER|>."},
        ],
        'arrangement': [
            {'role': 'user', 'content': "[QUALITY]: {'Average', 'High', 'Low'}\n[TOPIC]: is-porn-wrong\n[STANCE]: yes-porn-is-wrong\n[DEFINITION]: arrangement: Argumentation is arranged properly if it presents the issue, the arguments, and its conclusion in the right order\n[SENTENCE]: The argument that porn would expose one to the reality in life, and thus increasing one's maturity and prevent possible future sexual assaults, is totally crap. <br/> porn is definitely wrong. after a dosage, the guilt of wasting ur time, wasting energy, wasting effort and not devoting urself to another thing and regretting that you should not have started would almost kill you. trust me. <br/> its a road that shld not be taken, its as bad as smoking and dugs. i think the only reason its not banned in most developed countries is that they are facing an aging population.\n"},
            {'role': 'assistant', 'content': "<|ANSWER|>Low<|ANSWER|>."},
            {'role': 'user', 'content': "[QUALITY]: {'Average', 'High', 'Low'}\n[TOPIC]: personal-pursuit-or-advancing-the-common-good\n[STANCE]: personal-pursuit\n[DEFINITION]: arrangement: Argumentation is arranged properly if it presents the issue, the arguments, and its conclusion in the right order\n[SENTENCE]: In order to help advance a common good, you must first realize what is your common good. You have to know when you are that better person to then spread your good to other people. You can't just assume that you are a great person if you don't have anything to show it. If you desire to see a world full of all this goodness, YOU have to be that spark that starts the fire.\n"},
            {'role': 'assistant', 'content': "<|ANSWER|>Average<|ANSWER|>."},
            {'role': 'user', 'content': "[QUALITY]: {'Average', 'High', 'Low'}\n[TOPIC]: ban-plastic-water-bottles\n[STANCE]: yes-emergencies-only\n[DEFINITION]: arrangement: Argumentation is arranged properly if it presents the issue, the arguments, and its conclusion in the right order\n[SENTENCE]: Yes I do feel that the consumption of water bottles should not be allowed anywhere unless in the case of emergency. Plastic bottles can leak chemicals after a period of time. Water bottles also are almost never recycled, and end up in landfills which lead to pollution of our environment. They take 700 years to start to decompose. 90% of the cost is the bottle itself... The water is usually tap water, and is not regulated. Even if tap water is dirty, you can easily clean it out with leaves, moss, and some water cleanser. Nearly one in five tested water bottles have bacteria anyway.\n"},
            {'role': 'assistant', 'content': "<|ANSWER|>High<|ANSWER|>."},
        ],
        'clarity': [
            {'role': 'user', 'content': "[QUALITY]: {'Average', 'High', 'Low'}\n[TOPIC]: personal-pursuit-or-advancing-the-common-good\n[STANCE]: personal-pursuit\n[DEFINITION]: clarity: Argumentation has a clear style if it uses correct and widely unambiguous language as well as if it avoids unnecessary complexity and deviation from the issue\n[SENTENCE]: In order to help advance a common good, you must first realize what is your common good. You have to know when you are that better person to then spread your good to other people. You can't just assume that you are a great person if you don't have anything to show it. If you desire to see a world full of all this goodness, YOU have to be that spark that starts the fire.\n"},
            {'role': 'assistant', 'content': "<|ANSWER|>Low<|ANSWER|>."},
            {'role': 'user', 'content': "[QUALITY]: {'Average', 'High', 'Low'}\n[TOPIC]: personal-pursuit-or-advancing-the-common-good\n[STANCE]: personal-pursuit\n[DEFINITION]: clarity: Argumentation has a clear style if it uses correct and widely unambiguous language as well as if it avoids unnecessary complexity and deviation from the issue\n[SENTENCE]: Human nature is to endeavor personal pursuit. If you assume that human nature is to endeavor advancing the common good, then you're going to get screwed by someone who endeavors personal pursuit. Most Americans endeavor advancing the common good; which is why they get screwed by politicians who endeavor personal pursuit. If everyone endeavors personal pursuit, then we would keep each other in check. It is easier to get most people to endeavor personal pursuit than it is to get most people to endeavor advancing the common good; which is why communism failed.\n"},
            {'role': 'assistant', 'content': "<|ANSWER|>Average<|ANSWER|>."},
            {'role': 'user', 'content': "[QUALITY]: {'Average', 'High', 'Low'}\n[TOPIC]: india-has-the-potential-to-lead-the-world\n[STANCE]: yes-for\n[DEFINITION]: clarity: Argumentation has a clear style if it uses correct and widely unambiguous language as well as if it avoids unnecessary complexity and deviation from the issue\n[SENTENCE]: I believe that India is fast emerging as a hub for international trade and investment. India has provided a huge opportunity to enhance trade and investments in sectors such as mechanical and electrical engineering, food processing, automotive, tourism and banking among others. Almost every multinational company is focusing on India.\n"},
            {'role': 'assistant', 'content': "<|ANSWER|>High<|ANSWER|>."},
        ],
        'cogency': [
            {'role': 'user', 'content': "[QUALITY]: {'Average', 'High', 'Low'}\n[TOPIC]: personal-pursuit-or-advancing-the-common-good\n[STANCE]: personal-pursuit\n[DEFINITION]: cogency: An argument is cogent if it has acceptable premises that are relevant to its conclusion and that are sufficient to draw the conclusion\n[SENTENCE]: Human nature is to endeavor personal pursuit. If you assume that human nature is to endeavor advancing the common good, then you're going to get screwed by someone who endeavors personal pursuit. Most Americans endeavor advancing the common good; which is why they get screwed by politicians who endeavor personal pursuit. If everyone endeavors personal pursuit, then we would keep each other in check. It is easier to get most people to endeavor personal pursuit than it is to get most people to endeavor advancing the common good; which is why communism failed.\n"},
            {'role': 'assistant', 'content': "<|ANSWER|>Low<|ANSWER|>.\n"},
            {'role': 'user', 'content': "[QUALITY]: {'Average', 'High', 'Low'}\n[TOPIC]: india-has-the-potential-to-lead-the-world\n[STANCE]: yes-for\n[DEFINITION]: cogency: An argument is cogent if it has acceptable premises that are relevant to its conclusion and that are sufficient to draw the conclusion\n[SENTENCE]: I believe that India is fast emerging as a hub for international trade and investment. India has provided a huge opportunity to enhance trade and investments in sectors such as mechanical and electrical engineering, food processing, automotive, tourism and banking among others. Almost every multinational company is focusing on India.\n"},
            {'role': 'assistant', 'content': "<|ANSWER|>Average<|ANSWER|>."},
            {'role': 'user', 'content': "[QUALITY]: {'Average', 'High', 'Low'}\n[TOPIC]: if-your-spouse-committed-murder-and-he-or-she-confided-in-you-would-you-turn-them-in\n[STANCE]: yes\n[DEFINITION]: cogency: An argument is cogent if it has acceptable premises that are relevant to its conclusion and that are sufficient to draw the conclusion\n[SENTENCE]: Murder under any circumstance is not right. A person who commits the act knows the consequences, and they know that they are guilty. I wouldn't be able to look at my spouse the same ever again. I would not be able to live with the secret of a murder. <br/> So yea, i would turn them in.\n"},
            {'role': 'assistant', 'content': "<|ANSWER|>High<|ANSWER|>."},
        ],
        'effectiveness': [
            {'role': 'user', 'content': "[QUALITY]: {'Average', 'High', 'Low'}\n[TOPIC]: personal-pursuit-or-advancing-the-common-good\n[STANCE]: personal-pursuit\n[DEFINITION]: effectiveness: Argumentation is effective if it persuades the target audience of (or corroborates agreement with) the author's stance on the issue\n[SENTENCE]: Human nature is to endeavor personal pursuit. If you assume that human nature is to endeavor advancing the common good, then you're going to get screwed by someone who endeavors personal pursuit. Most Americans endeavor advancing the common good; which is why they get screwed by politicians who endeavor personal pursuit. If everyone endeavors personal pursuit, then we would keep each other in check. It is easier to get most people to endeavor personal pursuit than it is to get most people to endeavor advancing the common good; which is why communism failed.\n"},
            {'role': 'assistant', 'content': "<|ANSWER|>Low<|ANSWER|>."},
            {'role': 'user', 'content': "[QUALITY]: {'Average', 'High', 'Low'}\n[TOPIC]: india-has-the-potential-to-lead-the-world\n[STANCE]: yes-for\n[DEFINITION]: effectiveness: Argumentation is effective if it persuades the target audience of (or corroborates agreement with) the author's stance on the issue\n[SENTENCE]: I believe that India is fast emerging as a hub for international trade and investment. India has provided a huge opportunity to enhance trade and investments in sectors such as mechanical and electrical engineering, food processing, automotive, tourism and banking among others. Almost every multinational company is focusing on India.\n"},
            {'role': 'assistant', 'content': "<|ANSWER|>Average<|ANSWER|>."},
            {'role': 'user', 'content': "[QUALITY]: {'Average', 'High', 'Low'}\n[TOPIC]: if-your-spouse-committed-murder-and-he-or-she-confided-in-you-would-you-turn-them-in\n[STANCE]: yes\n[DEFINITION]: effectiveness: Argumentation is effective if it persuades the target audience of (or corroborates agreement with) the author's stance on the issue\n[SENTENCE]: Murder under any circumstance is not right. A person who commits the act knows the consequences, and they know that they are guilty. I wouldn't be able to look at my spouse the same ever again. I would not be able to live with the secret of a murder. <br/> So yea, i would turn them in.\n"},
            {'role': 'assistant', 'content': "<|ANSWER|>High<|ANSWER|>."},
        ],
        'global_acceptability': [
            {'role': 'user', 'content': "[QUALITY]: {'Average', 'High', 'Low'}\n[TOPIC]: personal-pursuit-or-advancing-the-common-good\n[STANCE]: personal-pursuit\n[DEFINITION]: global_acceptability: Argumentation is acceptable if the target audience accepts both the consideration of the stated arguments for the issue and the way they are stated\n[SENTENCE]: Human nature is to endeavor personal pursuit. If you assume that human nature is to endeavor advancing the common good, then you're going to get screwed by someone who endeavors personal pursuit. Most Americans endeavor advancing the common good; which is why they get screwed by politicians who endeavor personal pursuit. If everyone endeavors personal pursuit, then we would keep each other in check. It is easier to get most people to endeavor personal pursuit than it is to get most people to endeavor advancing the common good; which is why communism failed.\n"},
            {'role': 'assistant', 'content': "<|ANSWER|>Low<|ANSWER|>."},
            {'role': 'user', 'content': "[QUALITY]: {'Average', 'High', 'Low'}\n[TOPIC]: india-has-the-potential-to-lead-the-world\n[STANCE]: yes-for\n[DEFINITION]: global_acceptability: Argumentation is acceptable if the target audience accepts both the consideration of the stated arguments for the issue and the way they are stated\n[SENTENCE]: I believe that India is fast emerging as a hub for international trade and investment. India has provided a huge opportunity to enhance trade and investments in sectors such as mechanical and electrical engineering, food processing, automotive, tourism and banking among others. Almost every multinational company is focusing on India.\n"},
            {'role': 'assistant', 'content': "<|ANSWER|>Average<|ANSWER|>."},
            {'role': 'user', 'content': "[QUALITY]: {'Average', 'High', 'Low'}\n[TOPIC]: is-it-better-to-have-a-lousy-father-or-to-be-fatherless\n[STANCE]: fatherless\n[DEFINITION]: global_acceptability: Argumentation is acceptable if the target audience accepts both the consideration of the stated arguments for the issue and the way they are stated\n[SENTENCE]: I think in any case its better to have no father then to have a lousy father either way you still lose. When you have a father you want him to be the best he can be so you can learn from him as you grow up. If you were to have a lousy father then whats the point of having a father at all if he's just going to be lousy and he wouldn't teach you anything. It's always better to have a respectable father figure in your life then to have a lousy father figure.\n"},
            {'role': 'assistant', 'content': "<|ANSWER|>High<|ANSWER|>."},
        ],
        'global_relevance': [
            {'role': 'user', 'content': "[QUALITY]: {'Average', 'High', 'Low'}\n[TOPIC]: personal-pursuit-or-advancing-the-common-good\n[STANCE]: personal-pursuit\n[DEFINITION]: global_relevance: Argumentation is relevant if it contributes to the issue's resolution, i.e., if it states arguments or other information that help to arrive at an ultimate conclusion\n[SENTENCE]: Human nature is to endeavor personal pursuit. If you assume that human nature is to endeavor advancing the common good, then you're going to get screwed by someone who endeavors personal pursuit. Most Americans endeavor advancing the common good; which is why they get screwed by politicians who endeavor personal pursuit. If everyone endeavors personal pursuit, then we would keep each other in check. It is easier to get most people to endeavor personal pursuit than it is to get most people to endeavor advancing the common good; which is why communism failed.\n"},
            {'role': 'assistant', 'content': "<|ANSWER|>Low<|ANSWER|>."},
            {'role': 'user', 'content': "[QUALITY]: {'Average', 'High', 'Low'}\n[TOPIC]: india-has-the-potential-to-lead-the-world\n[STANCE]: yes-for\n[DEFINITION]: global_relevance: Argumentation is relevant if it contributes to the issue's resolution, i.e., if it states arguments or other information that help to arrive at an ultimate conclusion\n[SENTENCE]: I believe that India is fast emerging as a hub for international trade and investment. India has provided a huge opportunity to enhance trade and investments in sectors such as mechanical and electrical engineering, food processing, automotive, tourism and banking among others. Almost every multinational company is focusing on India.\n"},
            {'role': 'assistant', 'content': "<|ANSWER|>Average<|ANSWER|>."},
            {'role': 'user', 'content': "[QUALITY]: {'Average', 'High', 'Low'}\n[TOPIC]: ban-plastic-water-bottles\n[STANCE]: no-bad-for-the-economy\n[DEFINITION]: global_relevance: Argumentation is relevant if it contributes to the issue's resolution, i.e., if it states arguments or other information that help to arrive at an ultimate conclusion\n[SENTENCE]: Banning plastic bottled water would be a huge mistake in this very moment. More than a million people in the United States purchase bottled water every day which is helping the economy com out of this recession we are in. maybe not in a big way but every kid of help counts! Bottled water also only makes less then 1&#xof; the worlds wastes and can be recycled! According to the National Association for PET Container resources, PET water bottles are no the most recycled container in curb side programs by weight and by number! <br/>\n"},
            {'role': 'assistant', 'content': "<|ANSWER|>High<|ANSWER|>."},
        ],
        'global_sufficiency': [
            {'role': 'user', 'content': "[QUALITY]: {'Average', 'High', 'Low'}\n[TOPIC]: personal-pursuit-or-advancing-the-common-good\n[STANCE]: personal-pursuit\n[DEFINITION]: global_sufficiency: Argumentation is sufficient if it adequately rebuts those counter-arguments to it that can be anticipated\n[SENTENCE]: Human nature is to endeavor personal pursuit. If you assume that human nature is to endeavor advancing the common good, then you're going to get screwed by someone who endeavors personal pursuit. Most Americans endeavor advancing the common good; which is why they get screwed by politicians who endeavor personal pursuit. If everyone endeavors personal pursuit, then we would keep each other in check. It is easier to get most people to endeavor personal pursuit than it is to get most people to endeavor advancing the common good; which is why communism failed.\n"},
            {'role': 'assistant', 'content': "<|ANSWER|>Low<|ANSWER|>."},
            {'role': 'user', 'content': "[QUALITY]: {'Average', 'High', 'Low'}\n[TOPIC]: india-has-the-potential-to-lead-the-world\n[STANCE]: yes-for\n[DEFINITION]: global_sufficiency: Argumentation is sufficient if it adequately rebuts those counter-arguments to it that can be anticipated\n[SENTENCE]: I believe that India is fast emerging as a hub for international trade and investment. India has provided a huge opportunity to enhance trade and investments in sectors such as mechanical and electrical engineering, food processing, automotive, tourism and banking among others. Almost every multinational company is focusing on India.\n"},
            {'role': 'assistant', 'content': "<|ANSWER|>Average<|ANSWER|>."},
        ],
        'reasonableness': [
            {'role': 'user', 'content': "[QUALITY]: {'Average', 'High', 'Low'}\n[TOPIC]: personal-pursuit-or-advancing-the-common-good\n[STANCE]: personal-pursuit\n[DEFINITION]: reasonableness: Argumentation is reasonable if it contributes to the issue's resolution in a sufficient way that is acceptable to the target audience.\n[SENTENCE]: Human nature is to endeavor personal pursuit. If you assume that human nature is to endeavor advancing the common good, then you're going to get screwed by someone who endeavors personal pursuit. Most Americans endeavor advancing the common good; which is why they get screwed by politicians who endeavor personal pursuit. If everyone endeavors personal pursuit, then we would keep each other in check. It is easier to get most people to endeavor personal pursuit than it is to get most people to endeavor advancing the common good; which is why communism failed.\n"},
            {'role': 'assistant', 'content': "<|ANSWER|>Low<|ANSWER|>."},
            {'role': 'user', 'content': "[QUALITY]: {'Average', 'High', 'Low'}\n[TOPIC]: india-has-the-potential-to-lead-the-world\n[STANCE]: yes-for\n[DEFINITION]: reasonableness: Argumentation is reasonable if it contributes to the issue's resolution in a sufficient way that is acceptable to the target audience.\n[SENTENCE]: I believe that India is fast emerging as a hub for international trade and investment. India has provided a huge opportunity to enhance trade and investments in sectors such as mechanical and electrical engineering, food processing, automotive, tourism and banking among others. Almost every multinational company is focusing on India.\n"},
            {'role': 'assistant', 'content': "<|ANSWER|>Average<|ANSWER|>."},
            {'role': 'user', 'content': "[QUALITY]: {'Average', 'High', 'Low'}\n[TOPIC]: ban-plastic-water-bottles\n[STANCE]: yes-emergencies-only\n[DEFINITION]: reasonableness: Argumentation is reasonable if it contributes to the issue's resolution in a sufficient way that is acceptable to the target audience.\n[SENTENCE]: Yes I do feel that the consumption of water bottles should not be allowed anywhere unless in the case of emergency. Plastic bottles can leak chemicals after a period of time. Water bottles also are almost never recycled, and end up in landfills which lead to pollution of our environment. They take 700 years to start to decompose. 90% of the cost is the bottle itself... The water is usually tap water, and is not regulated. Even if tap water is dirty, you can easily clean it out with leaves, moss, and some water cleanser. Nearly one in five tested water bottles have bacteria anyway.\n"},
            {'role': 'assistant', 'content': "<|ANSWER|>High<|ANSWER|>."},
        ],
        'local_relevance': [
            {'role': 'user', 'content': "[QUALITY]: {\'Average\', \'High\', \'Low\'}\n[TOPIC]: tv-is-better-than-books\n[STANCE]: tv\n[DEFINITION]: local_relevance: A premise of an argument is relevant if it contributes to the acceptance or rejection of the argument\'s conclusion\n[SENTENCE]: you can get the ""Foxtel Go"" app on your Ipad so you can watch TV where ever and whenever you want.\n"},
            {'role': 'assistant', 'content': "<|ANSWER|>Low<|ANSWER|>."},
            {'role': 'user', 'content': "[QUALITY]: {'Average', 'High', 'Low'}\n[TOPIC]: personal-pursuit-or-advancing-the-common-good\n[STANCE]: personal-pursuit\n[DEFINITION]: local_relevance: A premise of an argument is relevant if it contributes to the acceptance or rejection of the argument's conclusion\n[SENTENCE]: Human nature is to endeavor personal pursuit. If you assume that human nature is to endeavor advancing the common good, then you're going to get screwed by someone who endeavors personal pursuit. Most Americans endeavor advancing the common good; which is why they get screwed by politicians who endeavor personal pursuit. If everyone endeavors personal pursuit, then we would keep each other in check. It is easier to get most people to endeavor personal pursuit than it is to get most people to endeavor advancing the common good; which is why communism failed.\n"},
            {'role': 'assistant', 'content': "<|ANSWER|>Average<|ANSWER|>."},
            {'role': 'user', 'content': "[QUALITY]: {'Average', 'High', 'Low'}\n[TOPIC]: india-has-the-potential-to-lead-the-world\n[STANCE]: yes-for\n[DEFINITION]: local_relevance: A premise of an argument is relevant if it contributes to the acceptance or rejection of the argument's conclusion\n[SENTENCE]: I believe that India is fast emerging as a hub for international trade and investment. India has provided a huge opportunity to enhance trade and investments in sectors such as mechanical and electrical engineering, food processing, automotive, tourism and banking among others. Almost every multinational company is focusing on India.\n"},
            {'role': 'assistant', 'content': "<|ANSWER|>High<|ANSWER|>."},
        ],
        'credibility': [
            {'role': 'user', 'content': "[QUALITY]: {'Average', 'High', 'Low'}\n[TOPIC]: personal-pursuit-or-advancing-the-common-good\n[STANCE]: personal-pursuit\n[DEFINITION]: credibility: Argumentation creates credibility if it conveys arguments and similar in a way that makes the author worthy of credence\n[SENTENCE]: Human nature is to endeavor personal pursuit. If you assume that human nature is to endeavor advancing the common good, then you're going to get screwed by someone who endeavors personal pursuit. Most Americans endeavor advancing the common good; which is why they get screwed by politicians who endeavor personal pursuit. If everyone endeavors personal pursuit, then we would keep each other in check. It is easier to get most people to endeavor personal pursuit than it is to get most people to endeavor advancing the common good; which is why communism failed.\n"},
            {'role': 'assistant', 'content': "<|ANSWER|>Low<|ANSWER|>."},
            {'role': 'user', 'content': "[QUALITY]: {'Average', 'High', 'Low'}\n[TOPIC]: india-has-the-potential-to-lead-the-world\n[STANCE]: yes-for\n[DEFINITION]: credibility: Argumentation creates credibility if it conveys arguments and similar in a way that makes the author worthy of credence\n[SENTENCE]: I believe that India is fast emerging as a hub for international trade and investment. India has provided a huge opportunity to enhance trade and investments in sectors such as mechanical and electrical engineering, food processing, automotive, tourism and banking among others. Almost every multinational company is focusing on India.\n"},
            {'role': 'assistant', 'content': "<|ANSWER|>Average<|ANSWER|>."},
            {'role': 'user', 'content': "[QUALITY]: {'Average', 'High', 'Low'}\n[TOPIC]: if-your-spouse-committed-murder-and-he-or-she-confided-in-you-would-you-turn-them-in\n[STANCE]: yes\n[DEFINITION]: credibility: Argumentation creates credibility if it conveys arguments and similar in a way that makes the author worthy of credence\n[SENTENCE]: As an ambitious, young person wanting to become a lawful, successful, homicide detective, I would not be lenient with any murderer in my midst. <br/> Hopefully, the murder wouldn't be the result of a pleasure/malicious-kill, so that the sentencing won't be as harsh, but nonetheless, all murderers must be tried. After all, hopefully my spouse will understand that having to live in hiding is basically the same as being in prison except much worse since there would be little chance for parole since they will have to live with the guilt and/or the fear of being caught for the rest of their lives.\n"},
            {'role': 'assistant', 'content': "<|ANSWER|>High<|ANSWER|>."},
        ],
        'emotional_appeal': [
            {'role': 'user', 'content': "[QUALITY]: {\'Average\', \'High\', \'Low\'}\n[TOPIC]: india-has-the-potential-to-lead-the-world\n[STANCE]: no-against\n[DEFINITION]: emotional_appeal: Argumentation makes a successful emotional appeal if it creates emotions in a way that makes the target audience more open to the author\'s arguments\n[SENTENCE]: When the U.S. fought the Brits for their independence, the Brits wore a red uniform. Every time the Americans saw a British soldier, they would yell, ""The red coats are coming, the red coats are coming."" <br/> If India ever tried to take over the U.S., Americans would be forced to yell, ""The red dots are coming, the red dots are coming."" ;)\n"},
            {'role': 'assistant', 'content': "<|ANSWER|>Low<|ANSWER|>."},
            {'role': 'user', 'content': "[QUALITY]: {'Average', 'High', 'Low'}\n[TOPIC]: personal-pursuit-or-advancing-the-common-good\n[STANCE]: personal-pursuit\n[DEFINITION]: emotional_appeal: Argumentation makes a successful emotional appeal if it creates emotions in a way that makes the target audience more open to the author's arguments\n[SENTENCE]: Human nature is to endeavor personal pursuit. If you assume that human nature is to endeavor advancing the common good, then you're going to get screwed by someone who endeavors personal pursuit. Most Americans endeavor advancing the common good; which is why they get screwed by politicians who endeavor personal pursuit. If everyone endeavors personal pursuit, then we would keep each other in check. It is easier to get most people to endeavor personal pursuit than it is to get most people to endeavor advancing the common good; which is why communism failed.\n"},
            {'role': 'assistant', 'content': "<|ANSWER|>Average<|ANSWER|>."},
            {'role': 'user', 'content': "[QUALITY]: {'Average', 'High', 'Low'}\n[TOPIC]: is-it-better-to-have-a-lousy-father-or-to-be-fatherless\n[STANCE]: fatherless\n[DEFINITION]: emotional_appeal: Argumentation makes a successful emotional appeal if it creates emotions in a way that makes the target audience more open to the author's arguments\n[SENTENCE]: Honestly, I'd rather be completely fatherless, not saying that not having is better than having one, albeit a lousy one. However, I do believe that you can grow up completely fatherless and without any major male role model and still be successful. Take myself for example; I lived the first thirteen years of my life without a father or any fatherly interaction, and those were the essential times a kid needed to be around his/her father, I turned out great(Self-opinion). So yes, as opposed to having a lousy father that would rub off on me and mold me into half a man, I would much rather being fatherless.\n"},
            {'role': 'assistant', 'content': "<|ANSWER|>High<|ANSWER|>."},
        ],
        'sufficiency': [
            {'role': 'user', 'content': "[QUALITY]: {'Average', 'High', 'Low'}\n[TOPIC]: personal-pursuit-or-advancing-the-common-good\n[STANCE]: personal-pursuit\n[DEFINITION]: sufficiency: An argument's premises are sufficient if, together, they give enough support to make it rational to draw its conclusion\n[SENTENCE]: Human nature is to endeavor personal pursuit. If you assume that human nature is to endeavor advancing the common good, then you're going to get screwed by someone who endeavors personal pursuit. Most Americans endeavor advancing the common good; which is why they get screwed by politicians who endeavor personal pursuit. If everyone endeavors personal pursuit, then we would keep each other in check. It is easier to get most people to endeavor personal pursuit than it is to get most people to endeavor advancing the common good; which is why communism failed.\n"},
            {'role': 'assistant', 'content': "<|ANSWER|>Low<|ANSWER|>."},
            {'role': 'user', 'content': "[QUALITY]: {'Average', 'High', 'Low'}\n[TOPIC]: india-has-the-potential-to-lead-the-world\n[STANCE]: yes-for\n[DEFINITION]: sufficiency: An argument's premises are sufficient if, together, they give enough support to make it rational to draw its conclusion\n[SENTENCE]: I believe that India is fast emerging as a hub for international trade and investment. India has provided a huge opportunity to enhance trade and investments in sectors such as mechanical and electrical engineering, food processing, automotive, tourism and banking among others. Almost every multinational company is focusing on India.\n"},
            {'role': 'assistant', 'content': "<|ANSWER|>Average<|ANSWER|>."},
            {'role': 'user', 'content': "[QUALITY]: {'Average', 'High', 'Low'}\n[TOPIC]: should-physical-education-be-mandatory-in-schools\n[STANCE]: yes\n[DEFINITION]: sufficiency: An argument's premises are sufficient if, together, they give enough support to make it rational to draw its conclusion\n[SENTENCE]: PE should be compulsory because it keeps us constantly fit and healthy. If you really dislike sports, then you can quit it when you're an adult. But when you're a kid, the best thing for you to do is study, play and exercise. If you prefer to be lazy and lie on the couch all day then you are most likely to get sick and unfit. Besides, PE helps kids be better at teamwork.\n"},
            {'role': 'assistant', 'content': "<|ANSWER|>High<|ANSWER|>."},
        ],
    }
    return examples.get(quality_dim)

def get_examples(task_names:str, quality_dim:str=None):
    """Get the few-shot examples for a specific task
    """
    d_examples = {
        'aduc': get_examples_aduc(),
        'claim_detection': get_examples_cd(),
        'evidence_detection': get_examples_ed(),
        'evidence_type': get_examples_et(),
        'fallacies': get_examples_fd(),
        'quality': get_examples_quality(quality_dim),
        'relation': get_examples_ar(),
        'stance_detection': get_examples_cd(),
    }
    return d_examples.get((task_names))

def few_shot_prompt(x, task_name: str):
    """Create the prompt for the few-shot
    """
    conv = x['conversations']
    if task_name != 'quality':
        example = get_examples(task_name)
    else:
        qual = [
            'overall_quality',
            'local_acceptability',
            'appropriateness',
            'arrangement',
            'clarity',
            'cogency',
            'effectiveness',
            'global_acceptability',
            'global_relevance',
            'global_sufficiency',
            'reasonableness',
            'local_relevance',
            'credibility',
            'emotional_appeal',
            'sufficiency',
        ]
        for q in qual:
            if q in conv[1].get('content'):
                example = get_examples('quality', q)
    for idx, i in enumerate(example):
        conv.insert(idx + 1, i)
    return conv

def load_model(model_for_task:str, model_to_use:str):
    """Load the model for inference

    Parameters
    ----------
    model_for_task : str
        task name among [aduc, claim_detection, evidence_detection, evidence_type, fallacies, relation, stance_detection, mt_ft]
    model_to_use : str
        one among [few-shot, zero-shot, merged, fine-tuned]

    Returns
    -------
    model and tokenizer
    """
    max_seq_lenght = 2048
    dtype = None
    gpu_mem_use = 0.6
    match model_to_use:
        case 'few-shot':
            load_in_4bit = True
            model_name = f'unsloth/Meta-Llama-3.1-8B-Instruct'
        case 'zero-shot':
            load_in_4bit = True
            model_name = f'unsloth/Meta-Llama-3.1-8B-Instruct'
        case 'merged':
            load_in_4bit = False
            model_name = get_merged_model()
        case 'fine-tuned':
            load_in_4bit = False
            model_name = get_ft_model(model_for_task)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_lenght,
        dtype=dtype,
        load_in_4bit=load_in_4bit,
        fast_inference=True,
        gpu_memory_utilization=gpu_mem_use,
    )
    return model, tokenizer

def get_data_for_task(task_name:str, s_file:dict):
    """Get the data for a specific task

    Parameters
    ----------
    task_name : str
        task name among [aduc, claim_detection, evidence_detection, evidence_type, fallacies, relation, stance_detection, mt_ft]
    s_file : dict
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

    Returns
    -------
    labels
        set of labels of the task
    tr_d
        Train data
    val_d
        Validation data
    test_d
        Test data
    change_lbl
        Task specific function to unifie the labels name
    """
    match task_name:
        case 'aduc':
            labels, tr_d, val_d, test_d = aduc.get_data(s_file)
            change_lbl = aduc.change_lbl
        case 'claim_detection':
            labels, tr_d, val_d, test_d = cd.get_data(s_file)
            change_lbl = cd.change_lbl
        case 'evidence_detection':
            labels, tr_d, val_d, test_d = ed.get_data(s_file)
            change_lbl = ed.change_lbl
        case 'evidence_type':
            labels, tr_d, val_d, test_d = et.get_data(s_file)
            change_lbl = et.change_lbl
        case 'fallacies':
            labels, tr_d, val_d, test_d = fd.get_data(s_file)
            change_lbl = fd.change_lbl
        case 'relation':
            labels, tr_d, val_d, test_d = arc.get_data(s_file)
            change_lbl = arc.change_lbl
        case 'stance_detection':
            labels, tr_d, val_d, test_d = sd.get_data(s_file)
            change_lbl = sd.change_lbl
        case 'quality':
            labels, tr_d, val_d, test_d = aq.get_data(s_file)
            change_lbl = aq.change_lbl
    return labels, tr_d, val_d, test_d, change_lbl

def get_dataset_for_task(
    task_name:str,
    tokenizer,
    s_file:dict,
    system_prompt:str,
    chat_template:str,
    few_shot:bool = False,
):
    """Get the Dataset for a specific task

    Parameters
    ----------
    task_name : str
        task name among [aduc, claim_detection, evidence_detection, evidence_type, fallacies, relation, stance_detection, mt_ft]
    tokenizer
    s_file : dict
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
        }_description_
    system_prompt : str
        system prompt for a specific task
    chat_template : str
        chat template for the Llama model
    few_shot : bool, optional
        set to true if doing inference in a few-shot training, by default False

    Returns
    -------
    labels
        set of labels of the task
    tr_d
        Train dataset
    val_d
        Validation dataset
    test_d
        Test dataset
    change_lbl
        Task specific function to unifie the labels name
    """
    labels, tr_d, val_d, test_d, change_lbl = get_data_for_task(
        task_name=task_name,
        s_file=s_file
    )
    if few_shot:
        tmp = test_d.apply(
            lambda x: few_shot_prompt(x, task_name),
            axis=1,
        )
        test_d['conversations'] = tmp
    train_set, val_set, test_set = prt.get_datasets(
        tokenizer=tokenizer,
        train=tr_d,
        val=val_d,
        test=test_d,
        chat_template=chat_template,
        sys_prt=system_prompt,
    )
    # tmp_set = test_set[:10]
    # tmp_set = pd.DataFrame().from_records(tmp_set)
    # test_set = Dataset.from_pandas(tmp_set)
    return labels, train_set, val_set, test_set, change_lbl

def get_metric_inference(
    change_lbl,
    task_name:str,
    test_result:pd.DataFrame,
    s_file:dict,
    ollama_inference:bool=False,
):
    """Get the metrics

    Parameters
    ----------
    change_lbl : _type_
        Task specific function to unifie the labels
    task_name : str
        task name among [aduc, claim_detection, evidence_detection, evidence_type, fallacies, relation, stance_detection, mt_ft]
    test_result : pd.DataFrame
        prediction made by the model
    s_file : dict
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
    ollama_inference : bool, optional
        set to true if inference was made using ollama, by default False

    Returns
    -------
    dict
        dictionary containing the metrics
    dict
        in the case of the fallacy task, dictionary containing the metrics
    """
    if not ollama_inference:
        if task_name == 'fallacies':
            metric_single, metric_multi = metrics.get_metrics(change_lbl,   test_result)
            plot.plot_metric(
                metric=metric_single,
                title=f'Scores single for {task_name}',
                file_metric=s_file.get('metric_single')
            )
            plot.plot_metric(
                metric=metric_multi,
                title=f'Scores Multi for {task_name}',
                file_metric=s_file.get('metric_multi')
            )
            return (metric_single, metric_multi)
        else:
            metric_single, _ = metrics.get_metrics(
                change_lbl,
                test_result,
                is_multi_lbl=False
            )
            plot.plot_metric(
                metric=metric_single,
                title=f'Scores for {task_name}',
                file_metric=s_file.get('metric_single')
            )
            return metric_single
    else:
        if task_name == 'fallacies':
            metric_single, metric_multi = metrics.get_metrics(change_lbl, test_result)
            plot.plot_metric(
                metric=metric_single,
                title=f'Scores Ollama Single for {task_name}',
                file_metric=s_file.get('metric_single')
            )
            plot.plot_metric(
                metric=metric_multi,
                title=f'Scores Ollama Multi for {task_name}',
                file_metric=s_file.get('metric_multi')
            )
            return (metric_single, metric_multi)
        else:
            metric_single, _ = metrics.get_metrics(change_lbl, test_result, is_multi_lbl=False)
            plot.plot_metric(
                metric=metric_single,
                title=f'Scores Ollama for {task_name}',
                file_metric=s_file.get('metric_single')
            )
            return metric_single

def inference_unsloth(
    task_name:str,
    model_for_task:str,
    model_to_use:str,
    model=None,
    tokenizer=None,
    few_shot:bool=False,
):
    """Model inference with unsloth

    Parameters
    ----------
    task_name : str
        inference on task name among [aduc, claim_detection, evidence_detection, evidence_type, fallacies, relation, stance_detection, mt_ft]
    model_for_task : str
        task name for which the model was fine-tuned among [aduc, claim_detection, evidence_detection, evidence_type, fallacies, relation, stance_detection, mt_ft]
    model_to_use : str
        one among [few-shot, zero-shot, merged, fine-tuned]
    model : optional
        by default None
    tokenizer : optional
        by default None
    few_shot : bool, optional
        set to true if inference in a few-shot settings, by default False

    Returns
    -------
    dict
        result of the inference
    """
    if model is None and tokenizer is None:
        print(f'#### Loading Model for {model_for_task} ####')
        model, tokenizer = load_model(model_for_task, model_to_use)
    print(f'#### Inference on {task_name}')
    chat_template = utils.get_templates()
    system_prompt = get_syst_prompt(task_name)
    time = datetime.datetime.now().strftime('%Y-%m-%d_%H_%M_%S')
    s_file = utils.get_savefile(
        task_name=task_name,
        spl_name='spl2',
        m_name='Meta-Llama-3.1-8B-Instruct',
        n_sample=4000,
        epoch=2,
        train_resp=f'_all_data_{model_to_use}_{model_for_task}',
        outputs_dir=f'./outputs/{task_name}',
        time=time
    )
    print(f'##### Load Data')
    labels, _, _, test_set, change_lbl = get_dataset_for_task(
        task_name=task_name,
        tokenizer=tokenizer,
        s_file=s_file,
        system_prompt=system_prompt,
        chat_template=chat_template,
        few_shot=few_shot,
    )
    print(f'##### Start inference')
    test_result = tr.test(
        model=model,
        tokenizer=tokenizer,
        data_test=test_set,
        labels=labels,
        result_file=s_file.get('test_result_file'),
        few_shot=few_shot
    )
    print(f'##### Metrics')
    metric = get_metric_inference(
        change_lbl,
        task_name=task_name,
        test_result=test_result,
        s_file=s_file,
    )
    return metric
    
def inference_ollama(
    ollama_model_name:str,
    model_to_use:str,
    task_name:str,
):
    """Inference with ollama

    Parameters
    ----------
    ollama_model_name : str
        name of the model
    model_to_use : str
        one among [few-shot, zero-shot, merged, fine-tuned]
    task_name : str
        inference on task name among [aduc, claim_detection, evidence_detection, evidence_type, fallacies, relation, stance_detection, mt_ft]

    Returns
    -------
    dict
        result of the inference
    """
    print(f'#### Inference on {task_name}')
    time = datetime.datetime.now().strftime('%Y-%m-%d_%H_%M_%S')
    s_file = utils.get_savefile(
        task_name=task_name,
        spl_name='spl2',
        m_name='Meta-Llama-3.1-8B-Instruct',
        n_sample=4000,
        epoch=2,
        train_resp=f'_ollama_all_data_{model_to_use}_{ollama_model_name}',
        outputs_dir=f'./outputs/test_{task_name}',
        time=time
    )
    print(f'##### Load Data')
    labels, _, _, test_d, change_lbl = get_data_for_task(
        task_name=task_name,
        s_file=s_file,
    )
    res = []
    # s = '<[|]ANSWER[|]>'
    s = r'<\|ANSWER\|>(.*?)<\|(ANSWER|eot_id)\|>'
    names_dataset = test_d['datasets']
    true_labels = test_d['answer']
    print(f'##### Start Inference')
    for i in test_d['conversations']:
        response = chat(model=ollama_model_name, messages=i)
        print(response['message']['content'])
        print(f'##############################')
        tmp = re.split(s, response['message']['content'])
        lbs = {lbl.lower() for lbl in labels}
        pred = [
            re.sub(r'[^\w\s_-]', '', i) 
            for i in tmp 
            if re.sub(r'[^\w\s_-]', '', i).lower() in lbs
        ]
        print(f'# Prediction : {pred}')
        res.append(pred)
    tmp_pred = [i if i != [] else ['Failed'] for i in res]
    pred_flat = list(itertools.chain.from_iterable(tmp_pred))
    d_res = {'names': names_dataset, 'pred': pred_flat, 'lbl': true_labels}
    df_res = pd.DataFrame(data=d_res)
    df_res.to_csv(
        s_file.get('test_result_file'),
        index=False
    )
    print(f'##### Metrics')
    metric = get_metric_inference(
        change_lbl,
        task_name=task_name,
        test_result=df_res,
        s_file=s_file,
        ollama_inference=True,
    )
    return metric

def inference_on_all_data(
    model_for_task:str = '',
    model_to_use:str = '',
    inference_method:str = '',
    few_shot:bool = False,
):
    """Perform inference on all task

    Parameters
    ----------
    model_for_task : str, optional
        task name for which the model was fine-tuned among [aduc, claim_detection, evidence_detection, evidence_type, fallacies, relation, stance_detection, mt_ft], by default ''
    model_to_use : str, optional
        one among [few-shot, zero-shot, merged, fine-tuned], by default ''
    inference_method : str, optional
        set to ollama if performing inference with ollam, by default ''
    few_shot : bool, optional
        set to true if performing inference in a few-shot setting, by default False

    Returns
    -------
    dict
        dictionary containing the result of the inference
    """
    all_result = {}
    task_list=[
        'aduc',
        'claim_detection',
        'evidence_detection',
        'evidence_type',
        'fallacies',
        'quality',
        'relation',
        'stance_detection'
    ]
    match inference_method:
        case 'ollama':
            ollama_model = f'unsloth_model_{model_for_task}'
            for task in task_list:
                result = inference_ollama(
                    ollama_model_name=ollama_model,
                    model_to_use=model_to_use,
                    task_name=task,
                )
                all_result.update({task: result})
        case _ :
            print(f'#### Loading Model for {model_for_task} ####')
            model, tokenizer = load_model(model_for_task, model_to_use)
            for task in task_list:
                result = inference_unsloth(
                    task_name=task,
                    model_for_task=model_for_task,
                    model_to_use=model_to_use,
                    model=model,
                    tokenizer=tokenizer,
                    few_shot=few_shot,
                )
                all_result.update({task: result})
    return all_result