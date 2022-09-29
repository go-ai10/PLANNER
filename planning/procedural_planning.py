import openai
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from sentence_transformers import util as st_utils
import json
from icecream import ic
import pandas as pd
import time
import random
import os
from tensorboardX import SummaryWriter
from datetime import datetime
from nltk.translate.bleu_score import sentence_bleu

import spacy
import wmd

import argparse

from transformers.pipelines import pipeline
from transformers import AutoModelForCausalLM, AutoTokenizer

# CMLE
from torch import nn
from torch.nn import CrossEntropyLoss
from ipm_func import *

# IRM
from torch import nn, optim, autograd

parser = argparse.ArgumentParser(description='Causal')

parser.add_argument('--dotrain', action='store_true', help='training')
parser.add_argument('--doeval', action='store_true', help='evaluating')
parser.add_argument('--dodemo', action='store_true', help='demo')
parser.add_argument('--switch_knowledge', action='store_true', help='demo')
parser.add_argument('--use_slide_window', action='store_true', help='demo')
parser.add_argument('--use_parsing', action='store_true', help='demo')
parser.add_argument('--deduplicate', action='store_true', help='demo')
parser.add_argument('--use_offline', action='store_true', help='demo')
parser.add_argument('--load_model_path', type=str, default="", help='load model')
parser.add_argument('--api_key', type=str, default="NA", help='api key')

parser.add_argument('--language_model_type', choices=['gpt-j-6B', 't5-11b', 'gpt2-1.5B', 'gpt2', 'gpt2-xl', 'gpt3', 'gpt_neo', 't5', 'bart', 'bert', 'roberta'], default="gpt2", help='choices')
parser.add_argument('--model_type', choices=['concept_knowledge', 'task_only_base', 'base', 'base_tune', 'standard_prompt', 'soft_prompt_tuning', 'chain_of_thought', 'chain_of_cause', 'cmle_ipm', 'cmle_epm', 'irm', 'vae_r', 'rwSAM', 'counterfactual_prompt'], help='choices')
parser.add_argument('--variant_type', choices=['wo_symbolic', 'wo_causality', 'full'], default='full', help='choices')
parser.add_argument('--intervene_type', choices=['conf', 'how' ,'goal', 'none'], default='none', help='choices')
parser.add_argument('--use_soft_prompt_tuning', action='store_true', help='soft_prompt_tuning')
parser.add_argument('--objective_summarization', action='store_true', help='objective_summarization')
parser.add_argument('--data_type', choices=['wikihow', 'robothow'], help='choices')

parser.add_argument('--n_tokens', type=int, default=20, help='n_tokens')
parser.add_argument('--init_from_vocab', action='store_true', help='demo')
parser.add_argument('--open_loop', action='store_true', help='open_loop')
parser.add_argument('--triplet_similarity_threshold', type=float, default=0.4)
parser.add_argument('--limit_num', type=int, default=50)

parser.add_argument('--debug_num', type=int, default=6)
parser.add_argument('--held_out_idx', type=int, default=50)
parser.add_argument('--max_tokens', type=int, default=30)
parser.add_argument('--train_batch_size', type=int, default=16)
parser.add_argument('--encode_batch_size', type=int, default=512)
parser.add_argument('--num_train_epochs', type=int, default=2)
parser.add_argument('--cut_threshold', type=float, default=0.6)
parser.add_argument('--weight_decay', type=float, default=0.01)
parser.add_argument('--learning_rate', type=float, default=0.01)
parser.add_argument('--lr_scheduler_type', type=str, default="linear")
parser.add_argument('--num_warmup_steps', type=int, default=0)
parser.add_argument('--max_train_steps', type=int, default=3)
parser.add_argument('--test_start_idx', type=int, default=200)

args = parser.parse_args()

all_model_type_list = ['concept_knowledge', 'task_only_base', 'base', 'base_tune', 'standard_prompt', 'soft_prompt_tuning', 'chain_of_thought', 'chain_of_cause', 'cmle_ipm', 'cmle_epm', 'irm', 'vae_r', 'rwSAM', 'counterfactual_prompt']

LMTYPE_TO_LMID = {
    "gpt3": "gpt3",
    "gpt-j-6B": "EleutherAI/gpt-j-6B",
    "gpt2-1.5B": "gpt2",
    "gpt2": "gpt2-large",
    "gpt2-xl": "gpt2-xl",
    "gpt_neo": "EleutherAI/gpt-neo-1.3B",
    "t5": "t5-3b",
    "bert": "bert-large-uncased",
    "roberta": "roberta-large",
    "bart": "facebook/bart-large"
}

method_type=args.model_type
task_path = args.data_type + "/" + method_type + "/" + datetime.now().strftime("%Y_%m_%d") + "/" + "demo_{}_{}_inter{}_var{}_heldout{}_oloop{}_".format(args.language_model_type, args.model_type, args.intervene_type, args.variant_type ,args.held_out_idx, args.open_loop) + datetime.now().strftime("%H%M%S")
task_log_dir = os.path.join("../log", task_path)
if not os.path.isdir(task_log_dir): os.makedirs(task_log_dir)
writer = SummaryWriter(task_log_dir)
GPU = 0



if torch.cuda.is_available():
    torch.cuda.set_device(GPU)
OPENAI_KEY = None  # replace this with your OpenAI API key, if you choose to use OpenAI API

# ======== Define hyperparameteres for plan generation
source = 'huggingface'  # select from ['openai', 'huggingface']
planning_lm_id = LMTYPE_TO_LMID[args.language_model_type] #'gpt2-large'  # see comments above for all options
translation_lm_id = 'stsb-roberta-large'  # see comments above for all options
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_STEPS = 20  # maximum number of steps to be generated
CUTOFF_THRESHOLD = 0.6 if args.data_type == "robothow" else args.cut_threshold # early stopping threshold based on matching score and likelihood score
P = 0.5  # hyperparameter for early stopping heuristic to detect whether Planning LM believes the plan is finished
BETA = 0.3  # weighting coefficient used to rank generated samples
if source == 'openai':
    openai.api_key = OPENAI_KEY
    sampling_params = \
            {
                "max_tokens": args.max_tokens,
                "temperature": 0.6,
                "top_p": 0.9,
                "n": 10,
                "logprobs": 1,
                "presence_penalty": 0.5,
                "frequency_penalty": 0.3,
                "stop": '\n'
            }
elif source == 'huggingface':
    if args.language_model_type in ["t5", "bart"]:
        sampling_params = \
                {
                    "min_length": 50,
                    "temperature": 0.1,
                    "top_p": 0.9,
                    "num_return_sequences": 10,
                    "repetition_penalty": 1.2,
                    'use_cache': True,
                    'output_scores': True,
                    'return_dict_in_generate': True,
                    'do_sample': True,
                }
    else:
        sampling_params = \
                {
                    "max_tokens": args.max_tokens,
                    "temperature": 0.1,
                    "top_p": 0.9,
                    "num_return_sequences": 10,
                    "repetition_penalty": 1.2,
                    'use_cache': True,
                    'output_scores': True,
                    'return_dict_in_generate': True,
                    'do_sample': True,
                }

total_score = {}
for model_type_item in all_model_type_list:
    total_score[model_type_item] = {"sentence-bleu": 0, "wmd": 0, "mover": 0, "rouge-1-f": 0, "rouge-1-p": 0, "rouge-1-r": 0, "bert-score-f": 0, "bert-score-p": 0, "bert-score-r": 0}
# ======== Causal Fine-tune GPT2
from torch.utils.data import Dataset, DataLoader, random_split, RandomSampler, SequentialSampler

def format_time(elapsed):
    return str(datetime.timedelta(seconds=int(round((elapsed)))))

class GPT2Dataset(Dataset):

    def __init__(self, task_list, action_list, tokenizer, gpt2_type="gpt2", max_length=768):

        self.tokenizer = tokenizer
        self.input_ids = []
        self.label_ids = []
        self.attn_masks = []
        self.task_input_ids = []
        self.task_attn_masks = []
        self.action_input_ids = []
        self.action_attn_masks = []
        for task_txt, action_txt_list in zip(task_list, action_list):
            # Get predicted all PAD, when use '<|startoftext|>' and '<|endoftext|>'
            if True:
                history_action_txt = ""
                for current_action_txt in str(action_txt_list).split("##STEP##"):
                    if current_action_txt == "": continue
                    history_action_txt += str(current_action_txt) + '.'
                # ic(history_action_txt)
                encodings_dict = tokenizer(str(task_txt)+ history_action_txt, truncation=True, max_length=max_length-(args.n_tokens if args.use_soft_prompt_tuning else 0), padding="max_length")
                if args.use_soft_prompt_tuning:
                    for i in range(20):
                        encodings_dict['input_ids'].insert(0, 50256)
                        encodings_dict['attention_mask'].insert(0, 1)
                self.input_ids.append(torch.tensor(encodings_dict['input_ids']))
                self.attn_masks.append(torch.tensor(encodings_dict['attention_mask']))
            else:
                history_action_txt = ""
                if args.objective_summarization:
                    task_encodings_dict = tokenizer(str(task_txt), truncation=True, max_length=max_length, padding="max_length")
                for current_action_txt in str(action_txt_list).split("##STEP##"):
                    if current_action_txt == "": continue
                    encodings_dict = tokenizer(str(task_txt) + history_action_txt, truncation=True, max_length=max_length-(args.n_tokens if args.use_soft_prompt_tuning else 0), padding="max_length")
                    if args.objective_summarization:
                        labels_encodings_dict = tokenizer(str(current_action_txt), truncation=True, max_length=max_length-(args.n_tokens if args.use_soft_prompt_tuning else 0), padding="max_length")
                    if args.use_soft_prompt_tuning:
                        for i in range(20):
                            encodings_dict['input_ids'].insert(0, 50256)
                            encodings_dict['attention_mask'].insert(0, 1)
                            if args.objective_summarization:
                                labels_encodings_dict['input_ids'].insert(0, 50256)
                                labels_encodings_dict['attention_mask'].insert(0, 1)
                    history_action_txt += str(current_action_txt) + '.'
                    self.input_ids.append(torch.tensor(encodings_dict['input_ids']))
                    self.attn_masks.append(torch.tensor(encodings_dict['attention_mask']))
                    if args.objective_summarization:
                        action_encodings_dict = tokenizer(history_action_txt, truncation=True, max_length=max_length, padding="max_length")
                        self.label_ids.append(torch.tensor(labels_encodings_dict['input_ids']))
                        self.task_input_ids.append(torch.tensor(task_encodings_dict['input_ids']))
                        self.task_attn_masks.append(torch.tensor(task_encodings_dict['attention_mask']))
                        self.action_input_ids.append(torch.tensor(action_encodings_dict['input_ids']))
                        self.action_attn_masks.append(torch.tensor(action_encodings_dict['attention_mask']))

            
    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        if args.objective_summarization:
            return self.input_ids[idx], self.label_ids[idx], self.attn_masks[idx], self.task_input_ids[idx], self.task_attn_masks[idx], self.action_input_ids[idx], self.action_attn_masks[idx]
        else:
            return self.input_ids[idx], self.attn_masks[idx], 

# ======== Soft-Prompt-Tuning https://github.com/kipgparker/soft-prompt-tuning
if args.language_model_type == "gpt3":
    import os
    import openai
    openai.api_key = args.api_key
    completion = openai.Completion()

    def ask_openloop(task_eval_predict, curr_prompt, model_type):
        if model_type == "task_only_base":
            prompt = task_eval_predict
        else:
            prompt = curr_prompt + task_eval_predict
        response = completion.create(
            prompt=prompt, engine="text-davinci-002", temperature=0.7,
            top_p=1, frequency_penalty=0, presence_penalty=0, best_of=1,
            max_tokens=120 if args.data_type == "wikihow" else args.max_tokens)
        answer = response.choices[0].text.strip().strip('-').strip('_')
        return answer.split('\n')

    def ask(prompt):
        import time
        time.sleep(3)
        try:
            response = completion.create(
                prompt=prompt, engine="text-davinci-002", temperature=0.7,
                top_p=1, frequency_penalty=0, presence_penalty=0, best_of=1,
                max_tokens=30)
        except:
            time.sleep(10)
            try:
                response = completion.create(
                    prompt=prompt, engine="text-davinci-002", temperature=0.7,
                    top_p=1, frequency_penalty=0, presence_penalty=0, best_of=1,
                    max_tokens=30)
            except:
                time.sleep(20)
                response = completion.create(
                    prompt=prompt, engine="text-davinci-002", temperature=0.7,
                    top_p=1, frequency_penalty=0, presence_penalty=0, best_of=1,
                    max_tokens=30)
        answer = response.choices[0].text.strip().strip('-').strip('_')
        ic(answer)
        return answer
model = None
if args.model_type in ["concept_knowledge", "task_only_base", "base", "standard_prompt", "chain_of_thought"]:
    if args.language_model_type in ["bert", "roberta", "gpt-j-6B", "gpt2-1.5B", "gpt2", "gpt2-xl", "gpt_neo"]:
        tokenizer = AutoTokenizer.from_pretrained(planning_lm_id)
        model = AutoModelForCausalLM.from_pretrained(planning_lm_id, pad_token_id=tokenizer.eos_token_id).to(device)
    elif args.language_model_type in ["bart"]:
        from transformers import BartForConditionalGeneration, BartTokenizer
        tokenizer = BartTokenizer.from_pretrained("facebook/bart-large")
        model = BartForConditionalGeneration.from_pretrained("facebook/bart-large", forced_bos_token_id=0)
    elif args.language_model_type in ["t5"]:
        from transformers import T5Tokenizer, T5ForConditionalGeneration
        tokenizer = T5Tokenizer.from_pretrained(planning_lm_id)
        model = T5ForConditionalGeneration.from_pretrained(planning_lm_id)
elif args.model_type in ["soft_prompt_tuning", "chain_of_cause", "cmle_ipm", "cmle_epm", "irm"]:
    from transformers import GPT2LMHeadModel, GPT2TokenizerFast
    from soft_embedding import SoftEmbedding
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    model = GPT2LMHeadModel.from_pretrained('gpt2', pad_token_id=tokenizer.eos_token_id)
    model.resize_token_embeddings(len(tokenizer))
    if args.use_soft_prompt_tuning:
        s_wte = SoftEmbedding(model.get_input_embeddings(), n_tokens=args.n_tokens, initialize_from_vocab=True)
        model.set_input_embeddings(s_wte)
    if args.load_model_path:
        model.load_state_dict(torch.load(args.load_model_path))
# ========
if args.objective_summarization:
    summarize_model = GPT2LMHeadModel.from_pretrained('gpt2', pad_token_id=tokenizer.eos_token_id)
    summarize_model.resize_token_embeddings(len(tokenizer))
    if torch.cuda.device_count() > 1:
        summarize_model = nn.DataParallel(summarize_model)
    summarize_model = summarize_model.to(device)
if torch.cuda.device_count() > 1:
    model= nn.DataParallel(model)
if model is not None:
    model = model.to(device)
# ======== Planning LM Initialization
def lm_engine(source, model, planning_lm_id, device):
    model = model

    def _generate(prompt, sampling_params):
        if source == 'openai':
            response = openai.Completion.create(engine=planning_lm_id, prompt=prompt, **sampling_params)
            generated_samples = [response['choices'][i]['text'] for i in range(sampling_params['n'])]
            # calculate mean log prob across tokens
            mean_log_probs = [np.mean(response['choices'][i]['logprobs']['token_logprobs']) for i in range(sampling_params['n'])]
        elif source == 'huggingface':
            if args.model_type in ["concept_knowledge", "task_only_base", "base", "standard_prompt", "chain_of_thought"]:
                if args.language_model_type == "bart":
                    prompt += ' <mask>'
                    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
                    prompt_len = input_ids.shape[-1]
                    if torch.cuda.device_count() > 1:
                        output_dict = model.module.generate(input_ids)
                    else:
                        output_dict = model.generate(input_ids)
                    return tokenizer.batch_decode(output_dict, skip_special_tokens=True)
                else:
                    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
                    prompt_len = input_ids.shape[-1]
                    if torch.cuda.device_count() > 1:
                        output_dict = model.module.generate(input_ids, max_length=prompt_len + args.max_tokens, **sampling_params)
                    else:
                        output_dict = model.generate(input_ids, max_length=prompt_len + args.max_tokens, **sampling_params)
            elif args.model_type in ["soft_prompt_tuning", "chain_of_cause", "cmle_ipm", "cmle_epm", "irm"]:
                inputs = tokenizer(prompt, return_tensors="pt")
                if args.use_soft_prompt_tuning:
                    inputs['input_ids'] = torch.cat([torch.full((1,args.n_tokens), 50256), inputs['input_ids']], 1).to(device)
                    inputs['attention_mask'] = torch.cat([torch.full((1,args.n_tokens), 1), inputs['attention_mask']], 1).to(device)
                inputs = inputs.to(device)

                prompt_len = inputs['input_ids'].shape[-1]
                sampling_params["use_cache"] = False
                if torch.cuda.device_count() > 1:
                    output_dict = model.module.generate(inputs['input_ids'], max_length=prompt_len + sampling_params['max_tokens'], **sampling_params)
                else:
                    # sampling_params['max_tokens']
                    output_dict = model.generate(inputs['input_ids'], max_length=prompt_len + 30, **sampling_params)

            # discard the prompt (only take the generated text)
            generated_samples = tokenizer.batch_decode(output_dict.sequences[:, prompt_len:])
            # calculate per-token logprob
            vocab_log_probs = torch.stack(output_dict.scores, dim=1).log_softmax(-1)  # [n, length, vocab_size]
            token_log_probs = torch.gather(vocab_log_probs, 2, output_dict.sequences[:, prompt_len:, None]).squeeze(-1).tolist()  # [n, length]
            # truncate each sample if it contains '\n' (the current step is finished)
            # e.g. 'open fridge\n<|endoftext|>' -> 'open fridge'
            for i, sample in enumerate(generated_samples):
                stop_idx = sample.index('\n') if '\n' in sample else None
                generated_samples[i] = sample[:stop_idx]
                token_log_probs[i] = token_log_probs[i][:stop_idx]
            # calculate mean log prob across tokens
            mean_log_probs = [np.mean(token_log_probs[i]) for i in range(sampling_params['num_return_sequences'])]
        generated_samples = [sample.strip().lower() for sample in generated_samples]
        return generated_samples, mean_log_probs

    return _generate

def mean_nll(logits, y):
    return nn.functional.binary_cross_entropy_with_logits(logits, y)

def IRM_penalty(logits, y):
    scale = torch.tensor(1.).cuda().requires_grad_()
    new_y = torch.tensor(np.zeros((y.size()[0], y.size()[1], 7), dtype='float'))
    for batch in range(new_y.size()[0]):
        for token in range(new_y.size()[1]):
            new_y[batch][token][y[batch][token]] = 1
    loss = mean_nll(logits * scale, new_y.to(device))
    grad = autograd.grad(loss, [scale], create_graph=True)[0]
    return torch.sum(grad**2)


if args.objective_summarization:
    generator = lm_engine(source, summarize_model, planning_lm_id, device)
else:
    generator = lm_engine(source, model, planning_lm_id, device)

# ======== Translation LM Initialization

# initialize Translation LM
translation_lm = SentenceTransformer(translation_lm_id).to(device)

if args.data_type == "robothow":
    atomic_action_list = ["walk to", "run to", "sit on", "stand up", "grab", "open", "close", "put", "put", "put", "put", "switch on", "switch off", "drink", "find", "pour", "type", "push", "pull", "wake up", "touch", "look at", "turn to", "rinse", "point at", "lie on", "wash", "squeeze", "eat", "scrub", "sleep", "cover", "climb", "watch", "plug in", "stir", "drop", "leave", "cut", "read", "spread", "plug out", "write", "uncover", "break", "spit", "wipe", "enter", "dance", "brush", "shake", "flush", "unfold", "put on", "put off", "unwrap", "stretch", "soak", "crawl", "wrap", "dial", "call", "fold", "throw", "crawl"]
    atomic_action_list_embedding = translation_lm.encode(atomic_action_list, batch_size=512, convert_to_tensor=True, device=device)

# create action embeddings using Translated LM
with open('../data/{}/{}_available_actions.json'.format(args.data_type, args.data_type), 'r') as f:
    action_list = json.load(f)
    if args.data_type == "robothow":
        action_list = [action.replace('_', ' ') for action in action_list]
action_list_embedding = translation_lm.encode(action_list, batch_size=512, convert_to_tensor=True, device=device)  # lower batch_size if limited by GPU memory


# create example task embeddings using Translated LM
available_examples_filepath = '../data/{}/{}_available_examples_inter{}_subset.json'.format(args.data_type, args.data_type, args.intervene_type) if not args.intervene_type == "none" else '../data/{}/{}_available_examples.json'.format(args.data_type, args.data_type)
with open(available_examples_filepath, 'r') as f:
    available_examples = json.load(f)

summarize_example_data_list = []
for example in available_examples[:args.held_out_idx]:
    summarize_example_data_list.append({"tasks": example.split('\n')[0], "steps": '.'.join(example.split('\n')[1:])})

# base exemplar and heldout
heldout_available_examples_filepath = '../data/{}/{}_available_examples.json'.format(args.data_type, args.data_type)
with open(heldout_available_examples_filepath, 'r') as f:
    heldout_available_examples = json.load(f)


heldout_example_task_list = [example.split('\n')[0] for example in heldout_available_examples[0 if args.model_type == "concept_knowledge" else -int(args.held_out_idx):]]
example_task_embedding = translation_lm.encode(heldout_example_task_list, batch_size=args.encode_batch_size, convert_to_tensor=True, device=device)  # lower batch_size if limited by GPU memory


# helper function for finding similar sentence in a corpus given a query
def find_most_similar(query_str, corpus_embedding):
    query_embedding = translation_lm.encode(query_str, convert_to_tensor=True, device=device)
    # calculate cosine similarity against each candidate sentence in the corpus
    cos_scores = st_utils.pytorch_cos_sim(query_embedding, corpus_embedding)[0].detach().cpu().numpy()
    # retrieve high-ranked index and similarity score
    most_similar_idx, matching_score = np.argmax(cos_scores), np.max(cos_scores)
    return most_similar_idx, matching_score

# ======== Autoregressive Plan Generation

result_list = []

nlp_encoder = spacy.load('en_core_web_md')

def calculate_total_score(total_score_cal, model_type, task_eval_groundtruth, task_eval_predict):
    import nltk
    task_eval_groundtruth = task_eval_groundtruth.replace('.', ' ')
    task_eval_predict = task_eval_predict.replace('.', ' ')
    total_score_cal[model_type]["sentence-bleu"] += nltk.translate.bleu_score.sentence_bleu([task_eval_groundtruth.split()], task_eval_predict.split())
    total_score_cal[model_type]["wmd"] += nlp_encoder(task_eval_groundtruth).similarity(nlp_encoder(task_eval_predict))

    from rouge import Rouge
    rouge = Rouge()
    scores = rouge.get_scores(task_eval_predict, task_eval_groundtruth)
    total_score_cal[model_type]["rouge-1-f"] += scores[0]["rouge-1"]["f"]
    total_score_cal[model_type]["rouge-1-p"] += scores[0]["rouge-1"]["p"]
    total_score_cal[model_type]["rouge-1-r"] += scores[0]["rouge-1"]["r"]
    from datasets import load_metric
    bertscore = load_metric("bertscore")
    try:
        bert_results = bertscore.compute(predictions=[task_eval_predict], references=[task_eval_groundtruth], model_type="distilbert-base-uncased")
        total_score_cal[model_type]["bert-score-f"] += bert_results["f1"][0]
        total_score_cal[model_type]["bert-score-p"] += bert_results["precision"][0]
        total_score_cal[model_type]["bert-score-r"] += bert_results["recall"][0]
    except:
        pass
    return total_score_cal

stage_depth = 0
def language_planning(total_score_cal, data_example, model_type, epoch_i=0):
    global example_task_embedding
    global action_list_embedding
    global result_list
    global generator
    global nlp_encoder

    task = data_example["tasks"]
    if model_type == "concept_knowledge":
        import requests
        limit = 10
        raw_concept_knowledge = set()
        concept_knowledge = {"pre":set(), "cur":set(), "sub":set(), "synonym":set()}
        loc_max_score_dict = {"pre":0, "cur":0, "sub":0}
        loc_max_dict = {"pre":"", "cur":"", "sub":""}
        use_max_score_dict = {"pre":0, "cur":0, "sub":0}
        use_max_dict = {"pre":"", "cur":"", "sub":""}

        cpnet_vocab_path = "../data/cpnet_subdata/concept.txt"
        with open(cpnet_vocab_path, "r", encoding="utf8") as fin:
            id2concept = [w.strip() for w in fin]
        concept2id = {w: i for i, w in enumerate(id2concept)}
        id2relation = conceptnet.merged_relations
        def concept_request(entity, stage):
            obj = requests.get('http://api.conceptnet.io/c/en/{}?limit={}'.format('_'.join(entity.lower().split()), limit)).json()
            obj.keys()
            global stage_depth
            global raw_concept_knowledge_str
            stage_depth += 1
            if stage_depth > 2: return
            for edge in obj['edges']:
                edge['end']['label'] = edge['end']['label'].replace(' it', ' ' + entity)
                edge['start']['label'] = edge['start']['label'].replace(' it', ' ' + entity)
                if edge['rel']['label'] in ['Synonym']:
                    concept_knowledge["synonym"].add(edge['end']['label'])

                if edge['rel']['label'] in ['HasPrerequisite']:
                    concept_request(edge['end']['label'], "pre")
                
                if edge['rel']['label'] in ['AtLocation']:
                    if not args.variant_type == "wo_causality":
                        node_embedding = translation_lm.encode(edge['end']['label'], convert_to_tensor=True, device=device)
                        task_embedding = translation_lm.encode(task[6:], convert_to_tensor=True, device=device)
                        # calculate cosine similarity against each candidate sentence in the corpus
                        cos_scores = st_utils.pytorch_cos_sim(node_embedding, task_embedding)[0].detach().cpu().numpy()
                        concept_knowledge[stage].add("Step: {}".format(edge['end']['label']))
                        if cos_scores > args.triplet_similarity_threshold or edge['weight'] > loc_max_score_dict[stage]:
                            concept_knowledge[stage].add("Step: {}".format(edge['start']['label']))
                    else:
                        if edge['weight'] > loc_max_score_dict[stage]:
                            loc_max_score_dict[stage] = edge['weight']
                            loc_max_dict[stage] = [edge['end']['label'], edge['start']['label']]

                if edge['rel']['label'] in ['UsedFor', 'RelatedTo']:
                    if not args.variant_type == "wo_causality":         
                        node_embedding = translation_lm.encode(edge['end']['label'], convert_to_tensor=True, device=device)
                        task_embedding = translation_lm.encode(task[6:], convert_to_tensor=True, device=device)
                        # calculate cosine similarity against each candidate sentence in the corpus
                        cos_scores = st_utils.pytorch_cos_sim(node_embedding, task_embedding)[0].detach().cpu().numpy()
                        if edge['weight'] > args.triplet_similarity_threshold or edge['weight'] > use_max_score_dict[stage]:
                            if edge['rel']['label'] in ['UsedFor']:
                                translated_atomic_idx, score = find_most_similar(edge['start']['label'] + edge['rel']['label'] + edge['end']['label'], atomic_action_list_embedding)
                                translated_atomic = atomic_action_list[translated_atomic_idx]
                                concept_knowledge[stage].add("Step: {} ".format(translated_atomic) + edge['start']['label'])
                        concept_knowledge[stage].add("Step: " + ' ' + edge['start']['label'] + edge['end']['label'])
                    else:
                        if edge['weight'] > use_max_score_dict[stage]:
                            use_max_score_dict[stage] = edge['weight']
                            use_max_dict[stage] = edge['start']['label']
                if not args.intervene_type in ['how', 'goal', 'conf']:
                    if len(loc_max_dict[stage]) > 0:
                        if args.variant_type == "wo_symbolic":
                            raw_concept_knowledge.append("Step: " + loc_max_dict[stage][0])
                            raw_concept_knowledge.append("Step: " + loc_max_dict[stage][1])
                        else:
                            translated_atomic_idx, score = find_most_similar(edge['start']['label'] + edge['rel']['label'] + edge['end']['label'], atomic_action_list_embedding)
                            translated_atomic = atomic_action_list[translated_atomic_idx]
                            concept_knowledge[stage].add("Step: {} ".format(translated_atomic) + loc_max_dict[stage][0])
                            concept_knowledge[stage].add("Step: {} ".format(translated_atomic) + loc_max_dict[stage][1])
                    if len(use_max_dict[stage]) > 0:
                        if args.variant_type == "wo_symbolic":
                            raw_concept_knowledge.append("Step: " + use_max_dict[stage])
                        else:
                            translated_atomic_idx, score = find_most_similar(edge['start']['label'] + edge['rel']['label'] + edge['end']['label'], atomic_action_list_embedding)
                            translated_atomic = atomic_action_list[translated_atomic_idx]
                            concept_knowledge[stage].add("Step: {} ".format(translated_atomic) + use_max_dict[stage])
        global stage_depth
        stage_depth = 0
        if not args.use_offline:
            concept_request(task[6:].lower().strip(), "cur")
            max_task_length = len(task[6:].split())
            limit_concept = max_task_length * args.debug_num
            if args.use_parsing:
                entity_list = get_atomic_tast_nodes(task)
                for entity in entity_list:
                    concept_request(entity.strip(), "cur")
            elif args.use_slide_window:
                for window_len in range(max_task_length-1 , 0, -1):
                    for start_idx in range(max_task_length-window_len+1):
                        if len(concept_knowledge["cur"]) + len(concept_knowledge["pre"]) + len(concept_knowledge["sub"]) >= limit_concept: break
                        task_atomic = task[6:].lower().split()[start_idx:start_idx+window_len]
                        task_atomic = ' '.join(task_atomic)
                        stage_depth = 0
                        concept_request(task_atomic.strip(), "cur")
            else:
                for task_atomic in task.split():
                    task_atomic = task_atomic.lower().strip()
                    stage_depth = 0
                    concept_request(task_atomic, "cur")
        else:
            get_offline_concept_knowledge(task)
        concept_list = []
        for key in concept_knowledge.keys():
            concept_list += [item for item in concept_knowledge[key]] 

        raw_concept_knowledge_str = f"{task}." + '.'.join(set(raw_concept_knowledge))

        translated_concept_str = f"{task}."
        translated_concept_list = []
        max_location_score = 0
        for step_idx, concept_step in enumerate(set(concept_list)):
            translated_concept_idx, score = find_most_similar(concept_step+' '+task[6:], action_list_embedding)
            translated_concept = action_list[translated_concept_idx]
            if args.data_type == "robothow" and any(loc in translated_concept for loc in ["livingroom","living room","bathroom","bedroom","dining room","home office","kitchen"]) and score > max_location_score:
                max_location_score = score
                if len(translated_concept_list) > 0 and any(loc in translated_concept_list[0] for loc in ["livingroom","living room","bathroom","bedroom","dining room","home office","kitchen"]):
                    translated_concept_list[0] = translated_concept
                else:
                    translated_concept_list.insert(0, translated_concept)
            else:
                translated_concept_list.append(translated_concept)
            
        dedup_translated_concept_list = list(set(translated_concept_list))
        if not args.variant_type == "wo_symbolic":
            dedup_translated_concept_list.sort(key=translated_concept_list.index)
        for step_idx, translated_concept in enumerate(dedup_translated_concept_list):
            translated_concept_str += f"\nStep {step_idx+1}: {translated_concept}."
        if args.variant_type == "wo_symbolic": curr_prompt = f'{raw_concept_knowledge_str}\n\n{task}.'
        if not args.open_loop:
            curr_prompt = f'{translated_concept_str}\n\n{task}.'
        else:
            curr_prompt = f'{translated_concept_str}'
    elif model_type == "chain_of_thought" or model_type == "chain_of_cause":
        exemplar_idx, _ = find_most_similar(task, chain_of_thought_task_embedding)
        exemplar = chain_of_thought_exemplar_list[exemplar_idx]
        example = chain_of_thought_program_list[exemplar_idx].replace('##STEP##', '\n')
        ic(exemplar_idx, exemplar, example)
        # construct initial prompt
        curr_prompt = f'{exemplar}{example}\n\n{task}.'
    elif model_type == "task_only_base":
        curr_prompt = task+'.' #+ "<|endoftext|>" 
    else:
        # find most relevant example        
        example_idx, _ = find_most_similar(task, example_task_embedding)
        example = heldout_available_examples[example_idx]
        ic(len(example_task_embedding), example)
        # construct initial prompt
        curr_prompt = f'{example}\n\n{task}.'
    

    result_list.append('\n' + '-'*10 + ' GIVEN EXAMPLE ' + '-'*10+'\n')
    task_eval_groundtruth = task + '. ' + str(data_example["steps"])
    result_list.append(task_eval_groundtruth)

    task_eval_predict = task + ". "
    task_prediced_steps = []
    for generator_item in [generator]:
        result_list.append(f'{task}.')
        step_sequence = []
        if args.language_model_type == "gpt3" and args.open_loop:
            generated_list = ask_openloop(task_eval_predict, curr_prompt, args.model_type)
            translated_list = []
            for step, each_step in enumerate(generated_list):
                most_similar_idx, matching_score = find_most_similar(each_step, action_list_embedding)
                translated_action = action_list[most_similar_idx]
                if matching_score < args.cut_threshold: continue
                best_action = translated_action
                previous_action = best_action
                formatted_action = (best_action[0].upper() + best_action[1:]).replace('_', ' ')
                step_sequence.append(translated_action)
                step_idx = step + 1
                translated_list.append(f' Step {step_idx}: {formatted_action}.')
            result_list.append(" ".join(translated_list.copy()))
            task_eval_predict += " ".join(translated_list.copy())
            ic(task_eval_predict)
        else:    
            for step in range(1, MAX_STEPS + 1):
                if args.language_model_type == "bart" or args.language_model_type == "gpt3":
                    sample = ask(curr_prompt) if args.language_model_type == "gpt3" else generator_item(curr_prompt + f'Step {step}: ', sampling_params)
                    most_similar_idx, matching_score = find_most_similar(sample, action_list_embedding)
                    translated_action = action_list[most_similar_idx]
                    if matching_score < args.cut_threshold: break
                    best_action = translated_action
                    previous_action = best_action
                    task_prediced_steps.append(best_action)
                    formatted_action = (best_action[0].upper() + best_action[1:]).replace('_', ' ')
                    curr_prompt += f'\nStep {step}: {formatted_action}.'
                    result_list.append(f'Step {step}: {formatted_action}.')
                    step_sequence.append(best_action)
                    task_eval_predict += f'Step {step}: {formatted_action}.'
                else:
                    best_overall_score = -np.inf
                    # query Planning LM for single-step action candidates
                    samples, log_probs = generator_item(curr_prompt + f'Step {step}: ', sampling_params) 
                    for sample, log_prob in zip(samples, log_probs):
                        most_similar_idx, matching_score = find_most_similar(sample, action_list_embedding)
                        overall_score = matching_score + BETA * log_prob
                        translated_action = action_list[most_similar_idx]
                        # heuristic for penalizing generating the same action as the last action
                        if step > 1 and translated_action == previous_action:
                            overall_score -= 0.5
                        # find the translated action with highest overall score
                        if overall_score > best_overall_score:
                            best_overall_score = overall_score
                            best_action = translated_action

                    top_samples_ids = np.argsort(log_probs)[-int(P * len(samples)):]
                    are_zero_length = all([len(samples[i]) == 0 for i in top_samples_ids])
                    below_threshold = best_overall_score < CUTOFF_THRESHOLD
                    if are_zero_length:
                        print(f'\n[Terminating early because top {P*100}% of samples are all 0-length]')
                        break
                    elif below_threshold:
                        print(f'\n[Terminating early because best overall score is lower than CUTOFF_THRESHOLD ({best_overall_score} < {CUTOFF_THRESHOLD})]')
                        break
                    else:
                        previous_action = best_action
                        task_prediced_steps.append(best_action)
                        formatted_action = (best_action[0].upper() + best_action[1:]).replace('_', ' ')
                        curr_prompt += f'\nStep {step}: {formatted_action}.'
                        result_list.append(f'Step {step}: {formatted_action}.')
                        step_sequence.append(best_action)
                        task_eval_predict += f'Step {step}: {formatted_action}.'
        if args.model_type == "concept_knowledge":
            if args.switch_knowledge:
                task_eval_predict = translated_concept_str.replace('\n', '. ') if not args.variant_type == "wo_symbolic" else raw_concept_knowledge_str
                result_list.append(task_eval_predict)
                result_list[-1:] = task_eval_predict
    return calculate_total_score(total_score_cal, model_type, task_eval_groundtruth, task_eval_predict)


if args.model_type == "chain_of_thought" or args.model_type == "chain_of_cause":
    f = open("../data/{}/{}_chain_of_thought.json".format(args.data_type, args.data_type))
    data = json.load(f)
    category_exemplars = data["exemplars"]["category_all"]
    chain_of_thought_task_list = [exemplar["task"] for exemplar in category_exemplars]
    chain_of_thought_exemplar_list = [exemplar["program_chain"] for exemplar in category_exemplars]
    chain_of_thought_task_embedding = translation_lm.encode(chain_of_thought_task_list, batch_size=args.encode_batch_size, convert_to_tensor=True, device=device)  # lower batch_size if limited 
    if args.model_type == "chain_of_thought":
        chain_of_thought_program_list = [exemplar["program_chain"] + "##PROGRAM##" + "##STEP##".join(exemplar["program"].split('\n')) for exemplar in category_exemplars]
    else:
        chain_of_thought_program_list = [exemplar["program_chain"] + exemplar["program_causal_chain"] + "##PROGRAM##" + "##STEP##".join(exemplar["program"].split('\n')) for exemplar in category_exemplars]

if args.dotrain:
    if args.data_type == "wikihow":
        df = pd.read_csv('../data/{}/wikicausal.csv'.format(args.data_type), delimiter='^')
        titles = df.title.copy()[args.test_start_idx:]
        headlines = df.headline.copy()[args.test_start_idx:]

        dataset = GPT2Dataset(titles, headlines, tokenizer, max_length=768)
    elif args.data_type == "robothow":
        f = open("../data/{}/{}_available_examples.json".format(args.data_type, args.data_type))
        data = json.load(f)
        example_action_list = ["##STEP##".join(example.split('\n')[1:]) for example in data[args.test_start_idx:]]  # following line contains actions
        dataset = GPT2Dataset(example_task_list, example_action_list, tokenizer, max_length=768)

    # Split into training and validation sets
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    # Create the DataLoaders for our training and validation datasets.
    # We'll take training samples in random order. 
    train_dataloader = DataLoader(
                train_dataset,  # The training samples.
                sampler = RandomSampler(train_dataset), # Select batches randomly
                batch_size = args.train_batch_size # Trains with this batch size.
            )

    # For validation the order doesn't matter, so we'll just read them sequentially.
    validation_dataloader = DataLoader(
                val_dataset, # The validation samples.
                sampler = SequentialSampler(val_dataset), # Pull out batches sequentially.
                batch_size = args.train_batch_size # Evaluate with this batch size.
            )
    model.cuda()

    seed_val = 42
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)

    epochs = args.num_train_epochs
    learning_rate = args.learning_rate
    warmup_steps = 1e2
    epsilon = 1e-8

    from transformers import AdamW, get_linear_schedule_with_warmup
    from torch import optim
    sample_every = 100
    total_steps = len(train_dataloader) * epochs

    if args.model_type in ["soft_prompt_tuning", "chain_of_thought", "chain_of_cause", "cmle_ipm", "cmle_epm", "irm"] and args.use_soft_prompt_tuning:
        #after updating the embedding, specify that you want to train the learned embedding
        if args.objective_summarization:
            optimizer = AdamW(summarize_model.parameters(), lr = learning_rate, eps = epsilon)
            scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps = warmup_steps, num_training_steps = total_steps)        
        else:
            optimizer = optim.Adam([model.module.transformer.wte.learned_embedding]) if torch.cuda.device_count() > 1 else optim.Adam([model.transformer.wte.learned_embedding])
            scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps = warmup_steps, num_training_steps = total_steps)
    else:
        optimizer = AdamW(model.parameters(), lr = learning_rate, eps = epsilon)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps = warmup_steps, num_training_steps = total_steps)


    total_t0 = time.time()

    training_stats = []

    model = model.to(device)

    for epoch_i in range(1, epochs+1):

        # ========================================
        #               Training
        # ========================================

        print("")
        print('======== Epoch {:} / {:} ========'.format(epoch_i, epochs))
        print('Training...')

        t0 = time.time()

        total_train_loss = 0

        
        if args.objective_summarization:
            summarize_model.train()
        else:
            model.train()
        accumulated_step = 0
        
        for step, batch in enumerate(train_dataloader):
            accumulated_step += 1
            b_input_ids = batch[0].to(device)
            b_labels = batch[0].to(device)
            b_masks = batch[1].to(device)
            if args.objective_summarization:
                b_task_input_ids = batch[3].to(device)
                b_task_masks = batch[4].to(device)
                b_action_input_ids = batch[5].to(device)
                b_action_masks = batch[6].to(device)

 
            if args.objective_summarization:
                summarize_model.zero_grad()   
            else:
                model.zero_grad()
            if args.model_type in ["soft_prompt_tuning", "chain_of_thought", "chain_of_cause", "cmle_ipm", "cmle_epm", "irm"]:
                # need to pad attention_mask and input_ids to be full seq_len + n_learned_tokens
                # even though it does not matter what you pad input_ids with, it's just to make HF happy
                # b_input_ids = torch.cat([torch.full((1,20), 50256), batch[0]], 1).to(device)
                # b_labels = torch.cat([torch.full((1,20), 50256), batch[0]], 1).to(device)
                # b_masks = torch.cat([torch.full((1,20), 1), batch[1]], 1).to(device)
                # ic(batch[0].shape, batch[1].shape, b_input_ids.shape, b_masks.shape)
                outputs = model(b_input_ids,
                                labels=b_labels, 
                                attention_mask = b_masks,
                                token_type_ids=None,
                                use_cache=False,
                                output_hidden_states=True if "cmle" in args.model_type else False
                                )
                if args.objective_summarization:
                    # ic(tokenizer.decode(outputs, skip_special_tokens=True))
                    # summarize_input = tokenizer.decode(outputs, skip_special_tokens=False)
                    summarize_outputs = summarize_model(b_action_input_ids,
                                    labels=b_task_input_ids,
                                    attention_mask = b_action_masks,
                                    token_type_ids=None,
                                    use_cache=False,
                                    output_hidden_states=True if "cmle" in args.model_type else False
                                    )
            else:
                outputs = model(b_input_ids,
                                labels=b_labels, 
                                attention_mask = b_masks,
                                token_type_ids=None
                                )
            if args.objective_summarization:
                loss = outputs[0].mean() + summarize_outputs[0].mean()
            else:
                loss = outputs[0].mean()
            batch_loss = loss.item()
            total_train_loss += batch_loss

            # Invariant Risk Minimization
            if args.model_type == "irm":
                irm_penalty = IRM_penalty(outputs[1], b_labels)
                total_train_loss += irm_penalty
                loss += irm_penalty

            # OOD regularizer: similar grad across domain_cls_loss / loss variance across domains (causal) domain generalization

            # CMLE Loss
            if args.model_type == "cmle_ipm":
                ipm_loss = 0.0
                encode_rep = outputs.hidden_states
                # encode_rep = torch.reshape(encode_rep, (encode_rep.size()[0], -1))
                # ic(outputs.logits.size()[1])
                # ic(tokenizer.decode(outputs, skip_special_tokens=True))
                # gpt2("Make dinner") => "text sequence xxxxxx" (batch_size, seq_length, vocab_size)
                # [sent1, ...,sent10] [ground_truth]
                # for action_label in range(action_labels):
                #     imb_dist, imb_mat = wasserstein(torch.tensor(encode_rep[0]), current_step_action_gt_label, dataset_action_prob_prior, action_label, device=device)
                # affordance: bread: {toastable, grabbable, eatable} remote control {grabbable, pushable}
                #"Task: Make Dinner; " => Step1: Go to the dinning room. Step 2: close the oven. Step 3: put food in the oven. Step 4: Close the oven."
                ipm_weight = 1
                for seq_l in range(50258): # seq_length 
                    # task encode_rep
                    imb_dist, imb_mat = wasserstein(torch.tensor(encode_rep[0]), torch.tensor([50257]), 1/50258, 50258, device=device)
                    ipm_loss += imb_dist * ipm_weight
                total_train_loss += ipm_loss
                loss += ipm_loss

            # Get sample every x batches.
            if step % sample_every == 0 and not step == 0:

                elapsed = format_time(time.time() - t0)
                print('  Batch {:>5,}  of  {:>5,}. Loss: {:>5,}.   Elapsed: {:}.'.format(step, len(train_dataloader), batch_loss, elapsed))
                writer.add_scalar('Step Training Batch Loss', batch_loss, epoch_i)

            loss.backward()
            optimizer.step()

            scheduler.step()

        # Calculate the average loss over all of the batches.
        avg_train_loss = total_train_loss / len(train_dataloader)       
        
        # Measure how long this epoch took.
        training_time = format_time(time.time() - t0)

        print("")
        print("  Average training loss: {0:.2f}".format(avg_train_loss))
        print("  Training epoch took: {:}".format(training_time))

        # ========================================
        #               Validation
        # ========================================

        print("")
        print("Running Validation...")

        t0 = time.time()

        model.eval()
        if args.objective_summarization:
            summarize_model.eval()

        total_eval_loss = 0
        nb_eval_steps = 0

        # Evaluate data for one epoch
        for batch in validation_dataloader:
            
            b_input_ids = batch[0].to(device)
            b_labels = batch[0].to(device)
            b_masks = batch[1].to(device)
            
            with torch.no_grad():        

                outputs  = model(b_input_ids, 
    #                            token_type_ids=None, 
                                attention_mask = b_masks,
                                labels=b_labels)
            
                loss = outputs[0].mean()  
                
            batch_loss = loss.item()
            total_eval_loss += batch_loss        

        avg_val_loss = total_eval_loss / len(validation_dataloader)
        
        validation_time = format_time(time.time() - t0)

        print("  Validation Loss: {0:.2f}".format(avg_val_loss))
        print("  Validation took: {:}".format(validation_time))

        # Record all statistics from this epoch.
        training_stats.append(
            {
                'epoch': epoch_i,
                'Training Loss': avg_train_loss,
                'Valid Loss': avg_val_loss,
                'Training Time': training_time,
                'Validation Time': validation_time
            }
        )
        writer.add_scalar('Training Loss', avg_train_loss, epoch_i)
        writer.add_scalar('Valid Loss', avg_val_loss, epoch_i)
        if torch.cuda.device_count() > 1:
            if args.objective_summarization:
                summarize_model_state_dict = summarize_model.module.state_dict()
            else:
                model_state_dict = model.module.state_dict()
        else:
            model_state_dict = model.state_dict()
        if epoch_i % 5 == 0:
            # eval during training
            total_score_cal = total_score.copy()
            for data_example in summarize_example_data_list:
                total_score_cal = language_planning(total_score_cal, data_example, args.model_type, epoch_i)
            writer.add_scalar('Valid Metric-BLEU', total_score_cal[args.model_type]["sentence-bleu"], epoch_i)
            writer.add_scalar('Valid Metric-WMD', total_score_cal[args.model_type]["wmd"], epoch_i)
            writer.add_scalar('Valid Metric-BERT', total_score_cal[args.model_type]["bert-score-f"], epoch_i)
            writer.add_scalar('Valid Metric-ROUGE', total_score_cal[args.model_type]["rouge-1-f"], epoch_i)
            ic(total_score_cal[args.model_type])
            task_checkpoints_dir = os.path.join("../checkpoints", task_path)
            if not os.path.isdir(task_checkpoints_dir) and args.dotrain: os.makedirs(task_checkpoints_dir)
            with open(os.path.join(task_checkpoints_dir, "config.json"), 'wt') as f:
                json.dump(vars(args), f, indent=4)
            torch.save(model_state_dict, os.path.join(task_checkpoints_dir, '{}_model_{}.pth'.format(args.model_type, epoch_i)))
            if args.objective_summarization:
                torch.save(summarize_model_state_dict, os.path.join(task_checkpoints_dir, '{}_summarize_model_{}.pth'.format(args.model_type, epoch_i)))

    print("")
    print("Training complete!")
    print("Total training took {:} (h:mm:ss)".format(format_time(time.time()-total_t0)))

    # ========

total_score_cal = total_score.copy()
task_result_dir = os.path.join("../result", task_path)
if not os.path.isdir(task_result_dir): os.makedirs(task_result_dir)
skip_count = 0
with open(os.path.join(task_result_dir, "{}_sum{}_task_result.txt".format(args.language_model_type, args.objective_summarization)), 'w') as resultfile:
    total_score_cal = total_score.copy()
    for data_example in summarize_example_data_list:
        
        total_score_cal = language_planning(total_score_cal, data_example, args.model_type)

    # mean value
    ic(len(summarize_example_data_list), total_score_cal[args.model_type].keys())
    for score_key in total_score_cal[args.model_type].keys():
        total_score_cal[args.model_type][score_key] /= (len(summarize_example_data_list)-skip_count)
    resultfile.writelines(result_list)
    json.dump(total_score_cal,resultfile)
    ic(skip_count, total_score_cal[args.model_type])
