# Readme.md

## Task

- Argumentative Discourse Unit Classification
  - [Microtext part 1](#argumentative-microtext-part-1)
  - [Microtext part 2](#argumentative-microtext-part-2)
  - [Perssuasive Essays](#persuassive-essays)
  - [AbstRCT](#abstrct)
- Claim Detection
  - [IAM Claim](#iam-claim-evidence-stance)
  - [IBM Claim](#ibm-claim)
  - [IBM Argument](#ibm-argument)
- Evidence Detection
  - [ArgSum Evidence cls](#argsum-dataset)
  - [IAM Evidence](#iam-claim-evidence-stance)
  - [IBM Evidence](#ibm-evidence)
- Argument Relation Classification
  - [Microtext part 1](#argumentative-microtext-part-1)
  - [Microtext part 2](#argumentative-microtext-part-2)
  - [Perssuasive Essays](#persuassive-essays)
  - [AbstRCT](#abstrct)
  - [Nixon-Kennedy Debates](#nixon-kennedy-debates)
  - [Node](#node)
  - [IBM Claim Polarity](#ibm-claim-polarity)
  - [ComArg](#comarg)
- Evidence Type Classification
  - [ArgSum Evidence Type](#argsum-dataset)
  - [IBM Type](#ibm-type)
  - [AQM](#aqm-dataset)
- Stance Detection
  - [IBM Claim Polarity](#ibm-claim-polarity)
  - [ComArg](#comarg)
  - [NLAS](#nlas-corpus)
  - [IAM Stance](#iam-claim-evidence-stance)
  - [FEVER](#fever)
  - [AQM](#aqm-dataset)
- Fallacies Detection
  - [CoCoLoFa](#cocolofa)
  - [MAFALDA](#mafalda)
- Argument Quality Assessment
  - [Dagstuhl-15512-argquality-corpus](#dagstuhl-15512-argquality-corpus)
- Argument Generation
  - [CounterArgGen](#counterarggen)
- Argument Summarization
  - [ArgSum Summary](#argsum-dataset)
  - [ConcluGen](#conclugen)
  - [DebateSum](#debatesum)
- Counter Speech Generation
  - [CounterSpeechDataset](#counter-speech-dataset)
- Enthymeme Reconstruction
  - [NLAS](#nlas-corpus)

## Datasets

### Argumentative Microtext (part 1)

Source :
> Andreas Peldszus and Manfred Stede. An annotated corpus of argumentative microtexts. In D. Mohammed, and M. Lewinski, editors, Argumentation and Reasoned Action - Proc. of the 1st European Conference on Argumentation, Lisbon, 2015. College Publications, London, 2016  
> xml data : [https://github.com/peldszus/arg-microtexts](https://github.com/peldszus/arg-microtexts) (data folder : corpus/en)  

Source json for ADU classification:  
> Abkenar, M. Y., Wang, W., Graupner, H., & Stede, M. (2024). Assessing Open-Source Large Language Models on Argumentation Mining Subtasks. arXiv preprint arXiv:2411.05639  
> json data : [https://anonymous.4open.science/r/openarg-41C0/README.md](https://anonymous.4open.science/r/openarg-41C0/README.md) (file : data/dfMT_PC.json)

Data in the *mt_p1.jsonl* under the following format:

- each line consist in a json containing the data of one microtext
- each json is under the following format:

```py
{  
    'id': name of the files (str),
    'topic': topic of the micro text (str),
    'stance' : stance toward the topic pro/con (str),
    'edu': [
      { id_edu (str) : text_edu (str) }
    ],
    'adu': [
      { 
        id_edu (str) : id_adu (str),
        'stance': pro/opp (str),
        'label': Premise/Claim (str)
      }
    ],
    'relations': [
      {
        'id': id_relations (str),
        'src': id_adu_src (str),
        'trg': id_trg (str),
        'type': reb/und/sup/add (str)
      }
    ]  
}
```

### Argumentative Microtext (part 2)

> Maria Skeppstedt, Andreas Peldszus and Manfred Stede. More or less controlled elicitation of argumentative text: Enlarging a microtext corpus via crowdsourcing. In Proc. 5th Workshop in Argumentation Mining (at EMNLP), Brussels, 2018  
> [https://github.com/discourse-lab/arg-microtexts-part2](https://github.com/discourse-lab/arg-microtexts-part2) (data folder : corpus)

Source json for ADU classification:  
> Abkenar, M. Y., Wang, W., Graupner, H., & Stede, M. (2024). Assessing Open-Source Large Language Models on Argumentation Mining Subtasks. arXiv preprint arXiv:2411.05639  
> json data : [https://anonymous.4open.science/r/openarg-41C0/README.md](https://anonymous.4open.science/r/openarg-41C0/README.md) (file : data/dfMTv2-PC.json)

Data in the *mt_p2.jsonl* under the following format:

- each line consist in a json containing the data of one microtext
- each json is under the following format:

```py
{  
    'id': name of the file (str),
    'topic': topic of the microtext (str),
    'stance': stance toward the topic pro/con (str),
    'edu': [ 
      { 
        id_edu (str) : text_edu (str)
      }
    ],
    'adu': [ 
      { 
        id_edu (str) : id_adu (str),
        'stance': pro/con (str),
        'label': Premise/Claim/Implicit Claim (str)
      }  
    ],
    'relations': [
      {
        'id' : id_relations (str),
        'src': id_adu_src (str),
        'trg': id_trg (str),
        'type': reb/und/sup/add (str)
      }  
    ]
}
```

### Persuassive Essays

Source :
> Stab, Christian; Gurevych, Iryna. Argument Annotated Essays (version 2). (2017). License description. Argument Mining, 4.43-06 Datenmanagement, datenintensive Systeme, Informatik-Methoden in der Wirtschaftsinformatik, 004. Technical University of Darmstadt.  
><https://tudatalib.ulb.tu-darmstadt.de/handle/tudatalib/2422>

Data in the *Persuasive_Essays.jsonl* under the following format:

- each line consist in a json containing on essays
- each json is under the following format:

```py
{  
    'id': name of the file (str),
    'text': text of the essays (str),
    'arguments': [
      {
        'id': id of the argument (str),
        'type': Major Claim/Claim/Premises (str),
        'text': text_argument (str),
        'span': (span_start (int), span_end (int))
      }  
    ],
    'relations': [
      {
        'id': id of the relation (str),
        'type': supports/attacks (str),
        'arg_src': id of the argument source of the relation (str),
        'arg_trg': id of the argument target of the relation (str)
      }
    ]  
}
```

### AbstRCT

Source :
> Tobias Mayer, Elena Cabrio and Serena Villata (2020) Transformer-based Argument Mining for Healthcare Applications.
In Proceedings of the 24th European Conference on Artificial Intelligence (ECAI 2020), Santiago de Compostela, Spain.  
>[https://gitlab.com/tomaye/abstrct/](https://gitlab.com/tomaye/abstrct/)  
> glaucoma test : AbstRCT_corpus/data/test/glaucoma_test  
> mixed test : AbstRCT_corpus/data/test/mixed_test  
> neoplasm data : AbstRCT_corpus/data/test/neoplasm_test, AbstRCT_corpus/data/dev/neoplasm_dev, AbstRCT_corpus/data/train/neoplasm_train

Data in *ABSTRCT_neo_jsonl* (neoplasm set), *ABSTRCT_test_glau.jsonl* (glaucoma test set) and *ABSTRCT_test_mixed.jsonl* (mixed between neo and glau set)  
Each file is under the following format:

- each line consist in a json containing the data of one element
- each json is under the following format:

```py
{  
    'id': name of the file (str),
    'text': entire text of the trial (str),
    'arguments': [
      {
        'id': id of the argument (str),
        'type': Major Claim/Claim/Premise (str),
        'text': text_argument (str),
        'span': (span_start (int), span_end (int)), 
      }  
    ],
    'relations': [
      {
        'id': id of the relation (str),
        'type': Attack/Support/Partial-Attack (str),
        'arg_src': id_argument_source (str),
        'arg_trg': id_argument_target (str)
      }
    ]
}
```

### CoCoLoFa

Source :
> Min-Hsuan Yeh, Ruyuan Wan, Ting-Hao Huang: CoCoLoFa: A Dataset of News Comments with Common Logical Fallacies Written by LLM-Assisted Crowds. EMNLP 2024: 660-677  
> [https://github.com/Crowd-AI-Lab/cocolofa/](https://github.com/Crowd-AI-Lab/cocolofa/)

Data in *cocolofa.jsonl* under the following format:

- each line consist of a json containing the data of one element
- each json is under the following format:

```py
{  
    'id': id of the news article (int),
    'title': title of the news article (str),
    'date': the publish date of the article (str),
    'author': author of the news article (str),
    'link': link to the article (str),
    'content': content of the article (str),
    'comments': [
      {
        'id ': id of the comment (int),
        'news_id': id of the corresponding article (int),
        'worker_id': id of the crowdworker (int),
        'respond_to': id of the comment that this comment respond to (int),
        'fallacy': logical fallacy contained in the comments (appeal to authority, appeal to majority, appeal to nature, appeal to tradition, appeal to worse problems, false dilemma, hasty generalization, slippery slope or none) (str),
        'comment': comment written by a crowdworker (str)
      },
    ]
}
```

### Nixon-Kennedy Debates

Source :
> Stefano Menini, Elena Cabrio, Sara Tonelli and Serena Villata. "Never retreat, Never retract: Argumentation analysis for political speeches". To appear in proceedings of the Thirty-Second AAAI Conference, New Orleans, Louisiana, USA, 2018  
> [https://dh.fbk.eu/2017/11/political-argumentation/](https://dh.fbk.eu/2017/11/political-argumentation/) (file : balanced_dataset.csv)

Data in *nixon_kennedy_debate.jsonl* under the following format:

- each lines consist in a json containing the data of one element
- each json is under the following format

```py
{
    'id': id of the argument pair (int),
    'topic': topic of the debates (str),
    'label': no_relation/support/attack (str),
    'argument1': sentence source of the relations (str),
    'argument3': sentence target of the relations (str)
}
```

### NoDe

Source:
> Cabrio, Elena, and Serena Villata. "Node: A benchmark of natural language arguments." Computational Models of Argument. IOS Press, 2014. 449-450.  
> [https://www-sop.inria.fr/NoDE/](https://www-sop.inria.fr/NoDE/)
> Debatepedia/ProCon dataset (files : procon.xml, debatepedia_train.xml, debatepedia_test.xml)
> 12angryman dataset (file : 12angry_man_final_dataset.xml)

Data in *node_debatepedia.jsonl* and *node_12angryman.jsonl* under the following format :

- each line consist in a json containing the data of one element
- each json is under the following format :

```py
# node_debatepedia.jsonl
{
    'topic': topic (str),
    'arg_src': source argument of the relation (str),
    'arg_trg': target argument of the relation (str),
    'label': attack/support (str)
}
# node_12angryman.jsonl
{
    'act': act of the 12 Angry Men play (str),
    'arg_src': source argument of the relation (str),
    'arg_trg': target argument of the relation (str),
    'label': attack/support (str)
}
```

### IBM-Claim-Polarity

Source :
> Roy Bar-Haim, Indrajit Bhattacharya, Francesco Dinuzzo, Amrita Saha, and Noam Slonim. 2017. Stance Classification of Context-Dependent Claims. In Proceedings of the 15th Conference of the European Chapter of the Association for Computational Linguistics: Volume 1, Long Papers. Association for Computational Linguistics, Valencia, Spain, pages 251–261.  
> [IBM Debater - Claim Stance Dataset](https://research.ibm.com/haifa/dept/vst/debating_data.shtml) (file : claim_stance_dataset_v1.json)

Data in *IBM_Claim_Polarity.jsonl* under the following format:

- each line consist of a json containing the data of one element
- each json is under the following format:

```py
{
    'topicId': topic id (int),
    'split': train/test (str),
    'topicText': text of the topic (str),
    'topicTarget': sentiment target of the topic (str),
    'topicSentiment': topic sentiment toward the target 1 (positive)/-1 (negative) (int),
    'claims': [
      {
        'claimId': claim id (int),
        'stance': PRO/CON (str),
        'claimCorrectedText': corrected text of the claim (str),
        'claimOriginalText': original text of the claim (str),
        'article': {
          'rawFile': path to the raw version of the article (str),
          'rawSpan': span of the claim in the raw version {
            'start': (int),
            'end': (int)
          },
          'cleanFile': path to the clean version of the article (str),
          'cleanSpan': span of the claim in the clean version {
            'start': (int),
            'end': (int)
          },
        },
        'compatible': yes/no (str) # see reference
        # if compatible == 'yes':
        'claimTarget': claim sentiment target {
          'text': (str),
          'span': {
            'start': (int),
            'end': (int)
          }
        },
        'claimSentiment':  sentiment of the claim toward the target 1 (positive) / -1 (negative) (int),
        'targetRelation': relation between claim target and topic target 1 (consistent) / -1 (contrastive) (int)
      }
    ]
}
```

### ComArg

Source :
> Filip Boltužić and Jan Šnajder (2014). Back up your Stance: Recognizing Arguments in Online Discussions. In Proceedings of the First Workshop on Argumentation Mining, Baltimore, Maryland. Association for Computational Linguistics, 49-58  
> [https://takelab.fer.hr/data/comarg/](https://takelab.fer.hr/data/comarg/)

Data in *ComArg.jsonl* under the following format:

- each line consist in a json containing the data of one element
- each json is under the following format:

```py
{
    'id': id of the argument (str),
    'topic': topic of the argument (str),
    'comment': {
      'text': text of the comment (str),
      'stance': Pro/Con (str)
    },
    'argument': {
      'text': text of the argument (str),
      'stance': Pro/Con (str)
    },
    'label': Attack/Implicit Attack, No use, Implicit Suppport, Support (str)
}
```

### NLAS Corpus

Source :
> Ramon Ruiz-Dolz, Joaquín Taverner, John Lawrence, Chris Reed: NLAS-multi: A Multilingual Corpus of Automatically Generated Natural Language Argumentation Schemes. CoRR abs/2402.14458 (2024)  
> [https://zenodo.org/records/8364002](https://zenodo.org/records/8364002) (only the english data)  

Additionnal source:
> Delas, Z., Plüss, B., & Ruiz-Dolz, R. (2024). An Argumentation Scheme-Based Framework for Automatic Reconstruction of Natural Language Enthymemes. In Computational Models of Argument (pp. 61-72). IOS Press  
> [https://github.com/zvonimir-delas/COMMA24-Enthymeme-Reconstruction](https://github.com/zvonimir-delas/COMMA24-Enthymeme-Reconstruction) (file : datasets/nlas_eng.json, datasets/nlas_extra_en.json)

Data in *nlas.jsonl* under the following format:

- each line consist in a json containing the data of one element:
- each json is under the following format:

```py
{
    'topic': topic of the argument (str),
    'stance': in favor/against (str),
    'argumentation scheme': argument scheme of the argument (str),
    'argument': argument presented according to the specified argumentation scheme (dict),
    'label': yes/no (str),
    'id': id of the data (str)
}
```

### Dagstuhl-15512-argquality-corpus

Source :
> Henning Wachsmuth, Nona Naderi, Yufang Hou, Yonatan Bilu, Vinodkumar Prabhakaran, Tim Alberdingk Thijm, Graeme Hirst, Benno Stein: Computational Argumentation Quality Assessment in Natural Language. EACL (1) 2017: 176-187  
> [https://zenodo.org/records/3973285](https://zenodo.org/records/3973285) (file : dagstuhl-15512-argquality-corpus-annotated.csv)

Data in *Dagstuhl_15512_argquality_corpus.jsonl* under the following format:

- each line consist of a json containing the data of one element
- each json is under the following format:

```py
{
    'id': id of the file (str),
    'topic': topic of the argument (str),
    'stance': stance of the argument (str),
    'is_argumentative': y/n (str),
    'argument': text of the argument (str),
    'label_quality': quality score according to the 15 dimensions [
        {
          'overall quality': {
            'annotators': list(str) of scores given by the 3 annotators (1 (Low)/ 2 (Average) / 3 (High)),
            'agg': mean of the scores given above (1 (Low)/ 2 (Average) / 3 (High)) (str),
          },
          'local acceptability': { ... },
          'appropriateness': { ... },
          'arrangement': { ... },
          'clarity': { ... },
          'cogency': { ... },
          'effectiveness': { ... },
          'global_acceptability': { ... },
          'global_relevance': { ... },
          'global_sufficiency': { ... },
          'reasonableness': { ... },
          'local_relevance': { ... },
          'credibility': { ... },
          'emotional_appeal': { ... },
          'sufficiency': { ... }
        }
    ]
}
```

### ArgSum Dataset

Source :
> Hao Li, Yuping Wu, Viktor Schlegel, Riza Batista-Navarro, Tharindu Madusanka, Iqra Zahid, Jiayan Zeng, Xiaochi Wang, Xinran He, Yizhi Li, Goran Nenadic: Which Side Are You On? A Multi-task Dataset for End-to-End Argument Summarisation and Evaluation. ACL (Findings) 2024: 133-150  
> [https://github.com/HaoBytes/ArgSum-Dataset](https://github.com/HaoBytes/ArgSum-Dataset)

Data in:

- *Argsum_evi_cls.jsonl* : data to classify sentences as Evidence or Irrelevant Evidence
- *Argsum_evi_type.jsonl*:  data to classify evidence according to their type
- *Argsum_summary.jsonl* : data to generate summary -> remove/rename element

under the following format:

- each line consist of a json containing the data of one element
- each json is under the following format:

```py
# Argsum_evi_cls
{
    'key_point_id': id (str),
    'evidence_id': id of the evidence (str),
    'evidence': text of the candidate evidence (str),
    'topic': topic of the argument (str),
    'source': source of the evidence (str),
    'argument': text of the argument (str),
    'stance': For/Against (str),
    'label': Evidence/Irrelevant Evidence (str)
}
# Argsum_evi_type
{
    'evidence_id': id of the evidence (str),
    'evidence': text of the evidence (str),
    'type': type of the evidence (str),
    'topic': topic (str),
    'source': source of the evidence (str)
}
# Argsum_summary
{
    'topic': topic (str),
    'stance': For/Against (str),
    'input': text to summarize (str)
    'summary': summary to generate (str)
}
```

### IAM-Claim-Evidence-Stance

Source :
> Liying Cheng, Lidong Bing, Ruidan He, Qian Yu, Yan Zhang, Luo Si: IAM: A Comprehensive and Large-Scale Dataset for Integrated Argument Mining Tasks. ACL (1) 2022: 2277-2287  
> [https://github.com/LiyingCheng95/IAM](https://github.com/LiyingCheng95/IAM)  
> claims data : claims/all_claims.txt  
> evidence data : evidence/evidence1.txt  
> stance data : stance/test.txt, stance/dev.txt, stance/train.txt

Data in:

- *iam_claim.jsonl* -> data for claim detection
- *iam_evidence.jsonl* -> data for evidence detection
- *iam_stance.jsonl* -> data for stance detection

under the following format:

- each line consist in a json containing the data of one element
- each json is under the following format:

```py
# iam_claim
{
    'label_claim': Claim/Non-claim (str),
    'topic': topic of the sentence (str),
    'sentence': sentence to classify (str),
    'article_id': id of the article (str),
    'label_stance': contest/support (str),
    'id': id of the element (str)
}
# iam_evidence
{
    'label_evidence': Evidence/Non-evidence (str),
    'claim': sentence of the claim (str),
    'evidence': sentence of the candidate evidence (str),
    'article_id': id of the article (str),
    'full_label': additionnal label for the pairs evidence/claim (str),
    'id': id of the element (str)
}
# iam_stance
{
    'label_claim': Claim/Non-claim (str),
    'topic': topic of the claim (str),
    'sentence': sentence of the claim (str),
    'article_id': id of the article (str),
    'label_stance': contest/support (str),
    'id': id of the element (str)
}
```

### IBM Claim

Source :
> Towards an argumentative content search engine using weak supervision Ran Levy, Ben Bogin, Shai Gretz, Ranit Aharonov and Noam Slonim COLING 2018  
> [IBM Debater - Claim Sentences Search](https://research.ibm.com/haifa/dept/vst/debating_data.shtml) (file : data_sets/test_set.csv)

Data in *ibm_claim.jsonl* under the following format:

- each line consist of a json containing the data of one element
- each json is under the following format:
  
```py
{
    'id': topic id (int),
    'topic': sentence of the topic (str),
    'sentence': sentence to classify (str),
    'label': Claim/Non-Claim (str),
    'url': link to the source (str)
}
```

### IBM Argument

Source :
> Unsupervised Expressive Rules Provide Explainability and Assist Human Experts Grasping New Domains Eyal Shnarch, Leshem Choshen, Guy Moshkowich, Noam Slonim and Ranit Aharonov Findings of the Association for Computational Linguistics: EMNLP 2020  
> [IBM Debater - Argumentative Sentences in Recorded Debates](https://research.ibm.com/haifa/dept/vst/debating_data.shtml)  

Data in *ibm_argument.jsonl* under the following format:

- each line consist of a json containing the data of one element
- each json is under the following format:

```py
{
    'sentence_id': id of the sentence (int),
    'sentence': sentence (str),
    'context': whole text containing the sentence (str),
    'topic': topic of the sentence (str),
    'label': Argument/Non Argument (str)
}
```

### IBM Evidence

Source :
> Will it Blend? Blending Weak and Strong Labeled Data in a Neural Network for Argumentation Mining Eyal Shnarch, Carlos Alzate, Lena Dankin, Martin Gleize, Yufang Hou, Leshem Choshen, Ranit Aharonov and Noam Slonim. Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics, 2018  
> [IBM Debater - Evidences Sentences](https://research.ibm.com/haifa/dept/vst/debating_data.shtml)

Data in *IBM_Evidence.jsonl* under the following format:

- each line consist in a json containing the data of one element
- each json is under the following format:

```py
{
    'topic': topic for the sentence candidate (str),
    'the concept of the topic': concept featured in the topic (str),
    'candidate': sentence of the evidence candidate (str),
    'candidate masked': sentence of the evidence candidate with the concept of the topic masked (str),
    'label': Evidence/Non-evidence (str),
    'wikipedia article name': title of the wikipedia article of the sentence candidate (str),
    'wikipedia url': url of the article (str)
}
```

### FEVER

Source :
> James Thorne, Andreas Vlachos, Christos Christodoulopoulos, Arpit Mittal: FEVER: a Large-scale Dataset for Fact Extraction and VERification. NAACL-HLT 2018: 809-819  
> [https://fever.ai/dataset/fever.html](https://fever.ai/dataset/fever.html) (file : train.jsonl, paper_dev.jsonl, paper_test.jsonl)

Data in *fever.jsonl* under the following format :

- each line consist in a json containing the data of one element
- each json is under the following format :

```py
{
    'id': id of the claim (int),
    'label': attack/support/neutral (str),
    'claim': sentences of the claim (str),
}
```

### IBM Type

Source :
> Ehud Aharoni, Anatoly Polnarov, Tamar Lavee, Daniel Hershcovich, Ran Levy, Ruty Rinott, Dan Gutfreund, Noam Slonim: A Benchmark Dataset for Automatic Detection of Claims and Evidence in the Context of Controversial Topics. ArgMining@ACL 2014: 64-68  
> [IBM Debater - Claims and Evidence](https://research.ibm.com/haifa/dept/vst/debating_data.shtml) (file : 2014_7_18_ibm_CDEdata.csv)

Data in *IBM_type.jsonl* under the following format:

- each line consist of a json containing the data of one element
- each json is under the following format:

```py
{
    'Topic': topic of the claim (str),
    'Article': name of the article (str),
    'Claim': sentence of the claim (str),
    'CDE': sentence of the evidence (str),
    'Type 1': type of the evidence (str),
    'Type 2': secondary type of the evidence (can be null) (str),
    'id': id of the element (str)
}
```

### AQM Dataset

Source :
> Jia Guo, Liying Cheng, Wenxuan Zhang, Stanley Kok, Xin Li, Lidong Bing: AQE: Argument Quadruplet Extraction via a Quad-Tagging Augmented Generative Approach. ACL (Findings) 2023: 932-946  
> [https://github.com/guojiapub/QuadTAG](https://github.com/guojiapub/QuadTAG) (data : data/QAM)

Data in *AQM.jsonl* under the following format:

- each line consist in a json containing the data of one element
- each json is under the following format:

```py
{
    'labels': [
      {
        'claim_idx': index of the claim sentence (int),
        'evidence_idx': index of the evidence sentence (int),
        'stance': For/Against (str),
        'evidence_type': type of the evidence sentence (str),
      },
    ]
    'sents': entire text split by sentence (list(str)),
    'doc_id': id of the document (str),
    'topic': topic of the text (str)
}
```

### CounterArgGen

Source :
> Milad Alshomary, Shahbaz Syed, Martin Potthast, Henning Wachsmuth: Argument Undermining: Counter-Argument Generation by Attacking Weak Premises. CoRR abs/2105.11752 (2021)  
> [https://github.com/webis-de/acl21-counter-argument-generation-by-attacking-weak-premises](https://github.com/webis-de/acl21-counter-argument-generation-by-attacking-weak-premises)  
> file : data/predictions/predictions_given_weak_premises/documents.json, data/predictions/predictions_given_weak_premises/references.txt

Data in *CounterArgGen.jsonl* under the following format:

- each line consist of a json containing the data of one element
- each json is under the following format:

```py
{
    'claim': text of the claim (str),
    'weak_premises': list(str) of the weak premise,
    'premises':list(str) of the different premise,
    'target': Counter arg to generate (str) 
}
```

### ConcluGen

Source :
> Shahbaz Syed, Khalid Al Khatib, Milad Alshomary, Henning Wachsmuth, Martin Potthast: Generating Informative Conclusions for Argumentative Texts. ACL/IJCNLP (Findings) 2021: 3482-3493  
> [https://huggingface.co/datasets/webis/conclugen](https://huggingface.co/datasets/webis/conclugen)

Data in :

- *ConcluGen_base.jsonl* -> argument and conclusion
- *ConcluGen_aspect.jsonl* -> argument,topic, aspect and conclusion
- *ConcluGen_target.jsonl* -> argument, topic, possible conclusion target and conclusion
- *ConcluGen_topic.jsonl* -> argument, topic and conclusion

under the following format:

- each line consist of a json containing the data of one element
- each json is under the following format

```py
# ConcluGen_base
{
    'argument': text of the argument (str),
    'conclusion': conclusion of the argument (str),
    'id': id of the argument (str)
}
# ConcluGen_aspect
{
    'argument': text of the argument (str),
    'conclusion': conclusion of the argument (str),
    'id': id of the argument (str)
    'topic': topic of the argument (str),
    'aspect': different aspect of the argument (str);
}
# ConcluGen_target
{
    'argument': text of the argument (str),
    'conclusion': conclusion of the argument (str),
    'id': id of the argument (str),
    'topic': topic of the argument (str),
    'target': possible conclusion target (str),
}
# ConcluGen_topic
{
    'argument': text of the argument (str),
    'conclusion': conclusion of the argument (str),
    'id': id of the argument (str),
    'topic': topic of the argument (str)
}
```

### DebateSum

Source :
> Allen Roush, Arvind Balaji: DebateSum: A large-scale argument mining and summarization dataset. CoRR abs/2011.07251 (2020)  
> [https://github.com/Hellisotherpeople/DebateSum](https://github.com/Hellisotherpeople/DebateSum)  
> [https://huggingface.co/datasets/Hellisotherpeople/DebateSum](https://huggingface.co/datasets/Hellisotherpeople/DebateSum)

Data in *DebateSum.jsonl* under the following format:

- each line consist of a json containing the data of one element
- each json is under the following format:

```py
{
    'Full-Document': text of the entire document (str),
    'Citation': Citation of the document (str),
    'Extract': Summary of the document (str),
    'Abstract': Short summary of the document (str),
    'id': id of the element (str)
}
```

### Counter-Speech Dataset

Source :
> Guizhen Chen, Liying Cheng, Anh Tuan Luu, Lidong Bing: Exploring the Potential of Large Language Models in Computational Argumentation. ACL (1) 2024: 2309-2330  
> [https://github.com/DAMO-NLP-SG/LLM-argumentation](https://github.com/DAMO-NLP-SG/LLM-argumentation) (data : sampled_data/counter_speech_generation/test)

Data in *CounterSpeechGen.jsonl* under the following format:

- each line consist of a json containing the data of one element
- each json is under the following format:

```py
{
    'speech_source': supporting speech about a topic (str),
    'speech_trg': counter speech about the same topic (str),
    'topic': topic of the speech (str)
}
```

### MAFALDA

Source :
> Chadi Helwe, Tom Calamai, Pierre-Henri Paris, Chloé Clavel, Fabian M. Suchanek: MAFALDA: A Benchmark and Comprehensive Study of Fallacy Detection and Classification. NAACL-HLT 2024: 4810-4845  
> [https://github.com/ChadiHelwe/MAFALDA](https://github.com/ChadiHelwe/MAFALDA) (file : datasets/gold_standard_dataset.jsonl)

Data in *mafalda.jsonl* under the following format:

- each line consist of a json containing the data of one element
- each json is under the following format:

```py
{
    'text': entire text (str),
    'labels': [
      list(str) of fallacy present in the text
    ],
    'comments': list(str) of details concerning the labels (str);
    'sentences_with_labels': [
      {
        'sentence': sentence extract from the text (str),
        'label': [
          list(str) of fallacy associated with the sentence (str)
        ]
      }
    ]
}
```
