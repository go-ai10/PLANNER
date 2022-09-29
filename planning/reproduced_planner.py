from unittest import result
import openai
import conceptnet
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from sentence_transformers import util as st_utils
import json
from icecream import ic
import pandas as pd
import time
import os
from tensorboardX import SummaryWriter
from datetime import datetime
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch import nn
from torch import nn, optim, autograd
from evaluation.metric import *
import random
from reward_func import *

from dataprocessor import *
from transformers import (
    MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING,
    WEIGHTS_NAME,
    AdamW,
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
    glue_compute_metrics as compute_metrics,
    glue_convert_examples_to_features as convert_examples_to_features,
    glue_output_modes as output_modes,
    glue_processors as processors,
)
parser = argparse.ArgumentParser(description='PromptPlanner')

parser.add_argument('--dotrain', action='store_true', help='training')
parser.add_argument('--doeval', action='store_true', help='evaluating')
parser.add_argument('--dodemo', action='store_true', help='demo')
parser.add_argument('--switch_knowledge', action='store_true', help='demo')
parser.add_argument('--use_slide_window', action='store_true', help='demo')
parser.add_argument('--use_indomain_data', action='store_true', help='use_indomain_data')
parser.add_argument('--deduplicate', action='store_true', help='demo')

parser.add_argument('--load_model_path', type=str, default="", help='load model')
parser.add_argument('--api_key', type=str, default="apikey", help='api key')
parser.add_argument('--summarize_intent', action='store_true', help='summarize_intent')
parser.add_argument('--summarize_action', action='store_true', help='summarize_action')
parser.add_argument('--use_intent_score', action='store_true', help='use_intent_score')
parser.add_argument('--language_model_type', choices=['gpt-j-6B', 't5-11b', 'gpt2-1.5B', 'gpt2', 'gpt2-xl', 'gpt3', 'gpt_neo', 't5', 'bart', 'bert', 'roberta'], default="gpt3", help='choices')
# base: zero-shot language planner
parser.add_argument('--model_type', choices=['intent_action_prompt', 'reward_action_prompt', 'intent_learning_prompt', 'all_learning_prompt', 'cpall_learning_prompt', 'action_prompt', 'cpnet_knowledge', 'ask_prompt', 'concept_knowledge', 'task_only_base', 'zs_planner', 'base_tune', 'standard_prompt', 'soft_prompt_tuning', 'chain_of_thought', 'chain_of_cause', 'cmle_ipm', 'cmle_epm', 'irm', 'vae_r', 'rwSAM', 'counterfactual_prompt'], help='choices', default='ask_prompt')
parser.add_argument('--action_prompt_mode_type', choices=['random', 'similarity'], default='similarity', help='choices')
parser.add_argument('--withoutconds', action='store_true', help='withoutconds')
parser.add_argument('--open_loop', action='store_true', help='open_loop')
parser.add_argument('--use_olddata', action='store_true', help='use_olddata')
parser.add_argument('--split_ask_prompt', action='store_true', help='split_ask_prompt')
parser.add_argument('--variant_type', choices=['wo_symbolic', 'wo_causality', 'full'], default='full', help='choices')
parser.add_argument('--intervene_type', choices=['conf', 'how' ,'goal', 'none'], default='none', help='choices')
parser.add_argument('--use_soft_prompt_tuning', action='store_true', help='soft_prompt_tuning')
parser.add_argument('--objective_summarization', action='store_true', help='objective_summarization')
parser.add_argument('--data_type', choices=['wikihow', 'robothow'], help='choices', default='robothow')
parser.add_argument('--split_type', choices=['train', 'dev', 'test'], help='choices', default='test')
parser.add_argument('--source_type', choices=['huggingface', 'openai'], help='choices', default='huggingface')


parser.add_argument('--reward_action_prompt_topk', type=int, default=20) # previous intent_action_prompt use 15
parser.add_argument('--reward_action_prompt_sample', type=int, default=20) # previous intent_action_prompt use 6
parser.add_argument('--max_stage_depth', type=int, default=20, help='max_stage_depth')
parser.add_argument('--n_tokens', type=int, default=20, help='n_tokens')

parser.add_argument('--action_trial_num', type=int, default=1)
parser.add_argument('--intent_trial_num', type=int, default=1)
parser.add_argument('--triplet_similarity_threshold', type=float, default=0.4)
parser.add_argument('--ratio_data', type=int, default=1)
parser.add_argument('--debug_num', type=int, default=6)
parser.add_argument('--held_out_idx', type=int, default=0)
parser.add_argument('--test_program_num', type=int, default=500)
parser.add_argument('--prompt_max_tokens', type=int, default=128)
parser.add_argument('--max_tokens', type=int, default=128)
parser.add_argument('--max_return_sample', type=int, default=1)
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

all_model_type_list = [args.model_type]

LMTYPE_TO_LMID = {
    # "gpt3": "gpt3",
    "gpt3": "text-davinci-002",
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

relation_groups = [
    'atlocation/locatednear',
    'capableof',
    'causes/causesdesire/*motivatedbygoal',
    'createdby',
    'desires',
    'antonym/distinctfrom',
    'hascontext',
    'hasproperty',
    'hassubevent/hasfirstsubevent/haslastsubevent/hasprerequisite/entails/mannerof',
    'isa/instanceof/definedas',
    'madeof',
    'notcapableof',
    'notdesires',
    'partof/*hasa',
    'relatedto/similarto/synonym',
    'usedfor',
    'receivesaction',
]

merged_relations = [
    'antonym',
    'atlocation',
    'capableof',
    'causes',
    'createdby',
    'isa',
    'desires',
    'hassubevent',
    'partof',
    'hascontext',
    'hasproperty',
    'madeof',
    'notcapableof',
    'notdesires',
    'receivesaction',
    'relatedto',
    'usedfor',
]

relation_text = [
    'is the antonym of',
    'is at location of',
    'is capable of',
    'causes',
    'is created by',
    'is a kind of',
    'desires',
    'has subevent',
    'is part of',
    'has context',
    'has property',
    'is made of',
    'is not capable of',
    'does not desires',
    'is',
    'is related to',
    'is used for',
]

method_type=args.model_type
# task_path = args.data_type + "/" + method_type + "/" + datetime.now().strftime("%Y_%m_%d") + "/" + "demo_{}_{}_inter{}_var{}_heldout{}_theta{}".format(args.language_model_type, args.model_type, args.intervene_type, args.variant_type, args.held_out_idx, args.cut_threshold) + datetime.now().strftime("%H%M")
task_path = args.data_type + "/" + method_type + "/" + "papertable_{}_{}_oloop{}_sum{}{}_sumaction{}{}_theta{}_wocond{}_indomain{}_n{}_mode{}".format(args.language_model_type, args.model_type, args.open_loop, args.summarize_intent, args.intent_trial_num, args.summarize_action, args.action_trial_num, args.cut_threshold, args.withoutconds, args.use_indomain_data, args.max_return_sample, args.action_prompt_mode_type) + datetime.now().strftime("%H%M")
# task_log_dir = os.path.join("../log", task_path)
# if not os.path.isdir(task_log_dir): os.makedirs(task_log_dir)

GPU = 0


if torch.cuda.is_available():
    torch.cuda.set_device(GPU)

source = args.source_type
planning_lm_id = LMTYPE_TO_LMID[args.language_model_type]
translation_lm_id = 'stsb-roberta-large'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_STEPS = 20
CUTOFF_THRESHOLD = args.cut_threshold
P = 0.5
BETA = 0.3
if source == 'openai':
    openai.api_key = args.api_key
    intent_prompt_sampling_params = \
            {
                "max_tokens": args.prompt_max_tokens,
                "temperature": 0.7,
                "top_p": 0.9,
                "n": 1,
                "logprobs": 1,
                "presence_penalty": 0.5,
                "frequency_penalty": 0.3
            }
    prompt_sampling_params = \
            {
                "max_tokens": args.prompt_max_tokens,
                "temperature": 0.7,
                "top_p": 0.9,
                "n": 1,
                "logprobs": 1,
                "presence_penalty": 0.5,
                "frequency_penalty": 0.3
            }
    sampling_params = \
            {
                "max_tokens": args.max_tokens,
                "temperature": 0.7, #0.6,
                "top_p": 1, #0.9,
                "n": args.max_return_sample,#5,
                "logprobs": 1,
                "presence_penalty": 0,#0.5,
                "frequency_penalty": 0,#0.3,
                # "stop": '\n'
            }
elif source == 'huggingface':
    intent_prompt_sampling_params = \
            {
                "max_tokens": args.prompt_max_tokens,
                "temperature": 0.1,
                "top_p": 0.9,
                "num_return_sequences": 1,
                "repetition_penalty": 1.2,
                'use_cache': True,
                'output_scores': True,
                'return_dict_in_generate': True,
                'do_sample': True,
            }
    prompt_sampling_params = \
            {
                "max_tokens": args.prompt_max_tokens,
                "temperature": 0.1,
                "top_p": 0.9,
                "num_return_sequences": 3,
                "repetition_penalty": 1.2,
                'use_cache': True,
                'output_scores': True,
                'return_dict_in_generate': True,
                'do_sample': True,
            }
    if args.language_model_type in ["t5", "bart"]:
        sampling_params = \
                {
                    "min_length": 50,
                    "temperature": 0.1,
                    "top_p": 0.9,
                    "num_return_sequences": 5,
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
    total_score[model_type_item] = {"sentence-bleu": 0, "wmd": 0, "sim": 0, "mover": 0, "rouge-1-f": 0, "rouge-1-p": 0, "rouge-1-r": 0, "bert-score-f": 0, "bert-score-p": 0, "bert-score-r": 0, "intent_score": 0}
from torch.utils.data import Dataset, DataLoader, random_split, RandomSampler, SequentialSampler

def format_time(elapsed):
    return str(datetime.timedelta(seconds=int(round((elapsed)))))

model = None

def model_to_category(model_type):
    if args.model_type in ["hitl", "reward_func", "intent_action_prompt", "reward_action_prompt", "intent_learning_prompt", "all_learning_prompt", "cpall_learning_prompt", "action_prompt", "ask_prompt", "concept_knowledge", "task_only_base", "zs_planner", "standard_prompt", "chain_of_thought"]:
        return "zeroshot"
    else:
        return "tuning"

if model_to_category(args.model_type) == "zeroshot":
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
elif model_to_category(args.model_type) == "tuning":
    from transformers import GPT2LMHeadModel, GPT2TokenizerFast
    # from soft_embedding import SoftEmbedding
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    model = GPT2LMHeadModel.from_pretrained('gpt2', pad_token_id=tokenizer.eos_token_id)
    model.resize_token_embeddings(len(tokenizer))
    if args.use_soft_prompt_tuning:
        s_wte = SoftEmbedding(model.get_input_embeddings(), n_tokens=args.n_tokens, initialize_from_vocab=True)
        model.set_input_embeddings(s_wte)
    if args.load_model_path:
        model.load_state_dict(torch.load(args.load_model_path))
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

def lm_engine(source, model, planning_lm_id, device):
    if torch.cuda.device_count() > 1:
        model= nn.DataParallel(model)
    else:
        model = model

    def _generate(prompt, sampling_params):
        if source == 'openai':
            response = openai.Completion.create(engine=planning_lm_id, prompt=prompt, **sampling_params)
            generated_samples = [response['choices'][i]['text'] for i in range(sampling_params['n'])]
            # ic(generated_samples, prompt, sampling_params)
            mean_log_probs = [np.mean(response['choices'][i]['logprobs']['token_logprobs']) for i in range(sampling_params['n'])]
        elif source == 'huggingface':
            if model_to_category(args.model_type) == "zeroshot":
                if args.language_model_type == "bart":
                    prompt += ' <mask>'
                    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
                    prompt_len = input_ids.shape[-1]
                    ic(prompt)
                    if torch.cuda.device_count() > 1:
                        output_dict = model.module.generate(input_ids)
                    else:
                        output_dict = model.generate(input_ids)
                    return tokenizer.batch_decode(output_dict, skip_special_tokens=True)
                else:
                    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
                    prompt_len = input_ids.shape[-1]
                    if torch.cuda.device_count() > 1:
                        output_dict = model.module.generate(input_ids, max_length=prompt_len + sampling_params['max_tokens'], **sampling_params)
                    else:
                        output_dict = model.generate(input_ids, max_length=prompt_len + sampling_params['max_tokens'], **sampling_params)
            elif model_to_category(args.model_type) == "tuning":
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
                    output_dict = model.generate(inputs['input_ids'], max_length=prompt_len + 30, **sampling_params)

            generated_samples = tokenizer.batch_decode(output_dict.sequences[:, prompt_len:])
            vocab_log_probs = torch.stack(output_dict.scores, dim=1).log_softmax(-1)
            token_log_probs = torch.gather(vocab_log_probs, 2, output_dict.sequences[:, prompt_len:, None]).squeeze(-1).tolist()
            for i, sample in enumerate(generated_samples):
                stop_idx = sample.index('\n') if '\n' in sample else None
                generated_samples[i] = sample[:stop_idx]
                token_log_probs[i] = token_log_probs[i][:stop_idx]
            mean_log_probs = [np.mean(token_log_probs[i]) for i in range(sampling_params['num_return_sequences'])]
        # ic(generated_samples)
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

translation_lm = SentenceTransformer(translation_lm_id).to(device)
if torch.cuda.device_count() > 1:
    translation_lm = nn.DataParallel(translation_lm).module

if args.use_olddata:
    with open('../data/{}/{}_available_actions.json'.format(args.data_type, args.data_type), 'r') as f:
        action_list = json.load(f)
        if args.data_type == "robothow":
            action_list = [action.replace('_', ' ') for action in action_list]
    action_list_embedding = translation_lm.encode(action_list, batch_size=512, convert_to_tensor=True, device=device)
else:
    actionset_path = '../data/{}_all/{}_program_actionset.json'.format(args.data_type, args.data_type) if args.data_type == "wikihow" else '../data/{}_all/{}_program_all_wocond{}_actionset.json'.format(args.data_type, args.data_type, args.withoutconds)
    with open(actionset_path, 'r') as f:
        action_list = json.load(f)
        if args.data_type == "robothow":
            action_list = [action.replace('_', ' ') for action in action_list]
    action_list_embedding = translation_lm.encode(action_list, batch_size=512, convert_to_tensor=True, device=device)

if args.model_type == "all_learning_prompt":
    atomic_action_list = action_list
    atomic_action_list_embedding = action_list_embedding
else:
    if args.data_type == "robothow":
        atomic_action_list = ["walk to", "run to", "sit on", "stand up", "grab", "open", "close", "put", "put", "put", "put", "switch on", "switch off", "drink", "find", "pour", "type", "push", "pull", "wake up", "touch", "look at", "turn to", "rinse", "point at", "lie on", "wash", "squeeze", "eat", "scrub", "sleep", "cover", "climb", "watch", "plug in", "stir", "drop", "leave", "cut", "read", "spread", "plug out", "write", "uncover", "break", "spit", "wipe", "enter", "dance", "brush", "shake", "flush", "unfold", "put on", "put off", "unwrap", "stretch", "soak", "crawl", "wrap", "dial", "call", "fold", "throw", "crawl"]
    elif args.data_type == "wikihow":
        atomic_actionset_path = '../data/{}_all/{}_program_atomic_actionset.json'.format(args.data_type, args.data_type)
        with open(atomic_actionset_path, 'r') as f:
            atomic_action_list = json.load(f)
    atomic_action_list_embedding = translation_lm.encode(atomic_action_list, batch_size=512, convert_to_tensor=True, device=device)

summarize_example_data_list = []
if args.use_olddata:
    available_examples_filepath = '../data/{}/{}_available_examples_inter{}_subset.json'.format(args.data_type, args.data_type, args.intervene_type) if not args.intervene_type == "none" else '../data/{}/{}_available_examples.json'.format(args.data_type, args.data_type)
    with open(available_examples_filepath, 'r') as f:
        available_examples = json.load(f)
    for example in available_examples[:args.held_out_idx]:
        summarize_example_data_list.append({"tasks": example.split('\n')[0], "steps": '.'.join(example.split('\n')[1:])})
else:
    import csv
    if args.data_type == "robothow":
        data_path = "/local/home/USERNAME/project/CAUSAL_PLANNER/data/{}_all/{}_program_all_wocond{}_{}.csv".format(args.data_type, args.data_type, args.withoutconds, args.split_type) if not args.use_indomain_data else "/local/home/USERNAME/project/CAUSAL_PLANNER/data/{}_all/{}_program_all_wocond{}_{}_indomain.csv".format(args.data_type, args.data_type, args.withoutconds, args.split_type)
    elif args.data_type == "wikihow":
        data_path = "/local/home/USERNAME/project/CAUSAL_PLANNER/data/{}_all/{}_program_catesample_{}.csv".format(args.data_type, args.data_type, args.split_type) if not args.use_indomain_data else "/local/home/USERNAME/project/CAUSAL_PLANNER/data/{}_all/{}_program_catesample_{}_indomain.csv".format(args.data_type, args.data_type, args.split_type)
    with open(data_path, newline='') as f:
        reader = csv.reader(f, delimiter='#')
        data = list(reader)
        for data_item in data[:min(len(data), 80)]:
            summarize_example_data_list.append({"tasks": data_item[0], "desp": data_item[1], "steps": '. '.join(data_item[2].split(';'))})
        summarize_example_data_list = summarize_example_data_list[1:args.test_program_num]
        # summarize_example_data_list = summarize_example_data_list[-args.test_program_num:]

heldout_available_examples_filepath = '../data/{}/{}_available_examples.json'.format(args.data_type, args.data_type)
with open(heldout_available_examples_filepath, 'r') as f:
    heldout_available_examples = json.load(f)

if args.use_olddata:
    ratio_data = args.ratio_data
    heldout_example_task_list = [example.split('\n')[0] for example in heldout_available_examples[0 if args.model_type == "concept_knowledge" else -int(args.held_out_idx/ratio_data):]]
else:
    all_examples_len = len(heldout_available_examples)
    ic(all_examples_len)
    # [:int(len(heldout_available_examples)/2)]
    if args.data_type == "robothow":
        heldout_available_examples = heldout_available_examples[:min(all_examples_len, 204)]
        heldout_example_task_list = [example.split('\n')[0] for example in heldout_available_examples]
    else:
        # heldout_available_examples = heldout_available_examples[900:1100]
        heldout_available_examples = heldout_available_examples[args.held_out_idx:args.held_out_idx+204]
        ic(len(heldout_available_examples))
        heldout_example_task_list = [example.split('\n')[0] for example in heldout_available_examples]
example_task_embedding = translation_lm.encode(heldout_example_task_list, batch_size=args.encode_batch_size, convert_to_tensor=True, device=device) 

def find_most_similar(query_str, corpus_embedding):
    try:
        query_embedding = translation_lm.encode(query_str, convert_to_tensor=True, device=device)
        cos_scores = st_utils.pytorch_cos_sim(query_embedding, corpus_embedding)[0].detach().cpu().numpy()
        most_similar_idx, matching_score = np.argmax(cos_scores), np.max(cos_scores)
        return most_similar_idx, matching_score
    except:
        return -1, -np.inf
    
def find_topk_similar(query_str, corpus_embedding, topk=10):
    try:
        query_embedding = translation_lm.encode(query_str, convert_to_tensor=True, device=device)
        cos_scores = st_utils.pytorch_cos_sim(query_embedding, corpus_embedding)[0].detach().cpu().numpy()
        # most_similar_idx, matching_score = np.argmax(cos_scores), np.max(cos_scores)
        most_similar_idxs = np.argpartition(cos_scores, -topk)[-topk:]
        matching_score = cos_scores[most_similar_idxs]
        return most_similar_idxs, matching_score
    except:
        return -1, -np.inf

result_list = []

stage_depth = 0

if args.model_type == "intent_learning_prompt" or args.model_type == "all_learning_prompt" or args.model_type == "cpall_learning_prompt":
    checkpoint = "/local/home/USERNAME/project/CAUSAL_PLANNER/output/{}_checkpoint/checkpoint-800/".format(args.data_type)
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
    model.to(device)
    reward_tokenizer = AutoTokenizer.from_pretrained(
        "bert-base-uncased",
        use_fast=False,
        # use_fast_tokenizer=False,
        # fast_tokenizer_class = None,
        do_lower_case=True,
        cache_dir=None,#args.cache_dir if args.cache_dir else None,
    )

    atomic_checkpoint = "/local/home/USERNAME/project/CAUSAL_PLANNER/output/{}_atomic_checkpoint/checkpoint-600/".format(args.data_type)
    atomic_model = AutoModelForSequenceClassification.from_pretrained(atomic_checkpoint)
    atomic_model.to(device)
    atomic_reward_tokenizer = AutoTokenizer.from_pretrained(
        "bert-base-uncased",
        use_fast=False,
        # use_fast_tokenizer=False,
        # fast_tokenizer_class = None,
        do_lower_case=True,
        cache_dir=None,#args.cache_dir if args.cache_dir else None,
    )

def hitl_cycle_score(task, task_eval_predict):
    global model
    if args.model_type == "zs_planner" or args.model_type == "intent_action_prompt" or args.model_type == "concept_knowledge" or args.model_type == "reward_action_prompt":
        look_forward_program = task_eval_predict
        ask_intent_prompt = look_forward_program + " Can you summarize my intent in 10 words?"

        response = openai.Completion.create(engine=planning_lm_id, prompt=ask_intent_prompt, **prompt_sampling_params)
        best_samples_intent, best_log_probs_intent = "", -np.inf
        samples_intent = [response['choices'][i]['text'] for i in range(prompt_sampling_params['n'])] 
        log_probs_intent = [np.mean(response['choices'][i]['logprobs']['token_logprobs']) for i in range(prompt_sampling_params['n'])]
        for sample, log_prob in zip(samples_intent, log_probs_intent):
            if log_prob > best_log_probs_intent:
                best_log_probs_intent = log_prob
                best_samples_intent = sample
        best_intent = best_samples_intent.strip().replace('\n', ' ')
        query_embedding = translation_lm.encode(task, convert_to_tensor=True, device=device)
        value_embedding = translation_lm.encode(best_intent, convert_to_tensor=True, device=device)
        matching_intent_score = calc_textemb_distance(query_embedding, value_embedding)
        return matching_intent_score
    elif args.model_type == "intent_learning_prompt" or args.model_type == "all_learning_prompt" or args.model_type == "cpall_learning_prompt":
        model = model.eval()
        # examples = ([task, task_eval_predict])
        examples = [InputExample(guid=0, text_a=task, text_b=task_eval_predict, label=3)]
        features = convert_examples_to_features(
            examples,
            reward_tokenizer,
            label_list=[3],
            max_length=128,
            output_mode="regression",
            # pad_on_left=bool(args.model_type in ["xlnet"]),  # pad on the left for xlnet
            # pad_token=tokenizer.pad_token_id,
            # pad_token_segment_id=tokenizer.pad_token_type_id,
        ) 
        with torch.no_grad():
            input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long).to(device)
            # reward_tokenizer([task, task_eval_predict], return_tensors="pt", padding=True).input_ids.to(device)
            # input_ids_b = reward_tokenizer(task_eval_predict, return_tensors="pt").input_ids.to(device)
            inputs = {"input_ids": input_ids}
            # ic(model(**inputs))
            logit = model(**inputs)["logits"]
        pred = np.squeeze(logit.detach().cpu().numpy())
        reward = pred
        # ic(reward)
        return reward
    else:
        return 1

def get_atomic_action_prompt(task, trial_num):
    global atomic_model
    if args.action_prompt_mode_type == "similarity":
        most_similar_idxs, matching_score = find_topk_similar(task, atomic_action_list_embedding, args.reward_action_prompt_topk)
        topk_list = [atomic_action_list[sim_idx] for sim_idx in most_similar_idxs]
    else:
        topk_list = atomic_action_list
    if args.model_type == "intent_learning_prompt":
        # if model_type == "intent_learning_prompt":
        #     checkpoint = "/local/home/USERNAME/project/CAUSAL_PLANNER/output/bert-base-uncased/STS-B/32_lang_percent100/checkpoint-800/"
        #     model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
        #     model.to(device)
        # else:
        loss_function = torch.nn.MSELoss()
        task_num_classes = 1
        reward_model = RewardActionFunction(
                            num_classes=task_num_classes,
                            loss_fn=loss_function,
                            prompt_feature_dim=1024,
                            task_feature_dim=1024,
                            fusion_output_size=1024,
                            dropout_p=0.1,
                            task_type="regression",
                        ).cuda(device)
        # similarity
        most_similar_idxs, matching_score = find_topk_similar(task, atomic_action_list_embedding, args.reward_action_prompt_topk)
        topk_list = [atomic_action_list[sim_idx] for sim_idx in most_similar_idxs]
        # sim_action_prompt = ', '.join()
        best_reward, best_trial = -np.inf, ""
        import time
        start = time.time()

        # reward_tokenizer = AutoTokenizer.from_pretrained(
        #     "bert-base-uncased",
        #     use_fast=False,
        #     # use_fast_tokenizer=False,
        #     # fast_tokenizer_class = None,
        #     do_lower_case=True,
        #     cache_dir=None,#args.cache_dir if args.cache_dir else None,
        # )

        for trial in range(trial_num):
            random_action_prompt = ', '.join(random.choices(topk_list, k=args.reward_action_prompt_sample))
            trial_prompt = f"Available actions are {random_action_prompt}, What are the steps needed to {task}."
            trial_prompt_emb = translation_lm.encode(trial_prompt, convert_to_tensor=True, device=device)
            task_emb = translation_lm.encode(task, convert_to_tensor=True, device=device)
            # if model_type == "intent_learning_prompt":
            #     with torch.no_grad():
            #         input_ids = reward_tokenizer(trial_prompt, return_tensors="pt").input_ids.to(device)
            #         inputs = {"input_ids": input_ids}
            #         logit = model(**inputs)["logits"]
            #         # ic(logit)
            # else:
            logit, loss = reward_model(trial_prompt_emb, task_emb)
            pred = np.squeeze(logit.detach().cpu().numpy())
            # ic(trial, pred)
            reward = pred
            if reward > best_reward:
                best_reward = reward
                best_trial = trial_prompt
        sampled_action_prompt = best_trial
    elif args.model_type == "all_learning_prompt":
        atomic_model = atomic_model.eval()
        most_similar_idxs, matching_score = find_topk_similar(task, atomic_action_list_embedding, args.reward_action_prompt_topk)
        topk_list = [atomic_action_list[sim_idx] for sim_idx in most_similar_idxs]
        best_reward, best_trial = -np.inf, ""

        for trial in range(trial_num):
            random_action_prompt = ', '.join(random.choices(topk_list, k=args.reward_action_prompt_sample))
            examples = [InputExample(guid=0, text_a=task, text_b=random_action_prompt, label=3)]
            features = convert_examples_to_features(
                examples,
                reward_tokenizer,
                label_list=[3],
                max_length=128,
                output_mode="regression",
                # pad_on_left=bool(args.model_type in ["xlnet"]),  # pad on the left for xlnet
                # pad_token=tokenizer.pad_token_id,
                # pad_token_segment_id=tokenizer.pad_token_type_id,
            ) 
            with torch.no_grad():
                input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long).to(device)
                inputs = {"input_ids": input_ids}
                logit = model(**inputs)["logits"]
            pred = np.squeeze(logit.detach().cpu().numpy())
            reward = pred
            if reward > best_reward:
                best_reward = reward
                trial_prompt = f"Available actions are {random_action_prompt}, What are the steps needed to {task}."
                best_trial = trial_prompt
        sampled_action_prompt = best_trial
    else:
        for trial in range(trial_num):
            sampled_action_prompt = ', '.join(random.choices(topk_list, k=args.reward_action_prompt_sample))
            action_confirm_prompt = f"Can you use these actions: {sampled_action_prompt}, for the task make drink? Yes or No?"
            response = openai.Completion.create(engine=planning_lm_id, prompt=action_confirm_prompt, **prompt_sampling_params)
            action_sample = response['choices'][0]['text']
            # ic(sampled_action_prompt, action_sample)
            if "no" in action_sample or "No" in action_sample: continue
    return sampled_action_prompt

def get_current_prompt(model_type, task):
    # action_prompt random sample (coverage) and permutation (order): structure prompt generation
    # no training, directly use intent score as reward function to choose structure prompt
    curr_prompt = []
    if model_type == "intent_action_prompt":
        # random_action_prompt_a = ', '.join(random.choices(atomic_action_list, k=10))
        # random_action_prompt_b = ', '.join(random.choices(atomic_action_list, k=10))
        # action_compare_prompt = f"Actions list a: {random_action_prompt_a}. Actions list b: {random_action_prompt_b}. Is Action list a or Action list b more related to the task {task}?"
        # response = openai.Completion.create(engine=planning_lm_id, prompt=action_compare_prompt, **prompt_sampling_params)
        # action_sample = response['choices'][0]['text']
        trial_num = args.action_trial_num if args.summarize_action else 1
        prompt_sample_num = args.intent_trial_num if args.summarize_intent else 1 
        for _ in range(prompt_sample_num):
            sampled_action_prompt = get_atomic_action_prompt(task, trial_num)
            curr_prompt.append(f"Available actions are {sampled_action_prompt}, What are the steps needed to {task}.")
    # use intent score to guide the reward function training: R(action_prompt, task) ~ Intent_Score(generated program)
    # elif model_type == "reward_action_prompt" or model_type == "intent_learning_prompt" or model_type == "all_learning_prompt":
    elif model_type == "reward_action_prompt" or model_type == "all_learning_prompt":
        trial_num = args.action_trial_num if args.summarize_action else 1
        prompt_sample_num = args.intent_trial_num if args.summarize_intent else 1 
        for _ in range(prompt_sample_num):
            sampled_action_prompt = get_atomic_action_prompt(task, trial_num)
            curr_prompt.append(f"Available actions are {sampled_action_prompt}, What are the steps needed to {task}.")
        # curr_prompt.append(best_trial)
    # intent_learning_prompt: w/o action reward learning, also directly use ava_actions
    elif model_type == "action_prompt" or model_type == "intent_learning_prompt":
        ava_actions = ",".join(random.choices(atomic_action_list, k=args.reward_action_prompt_sample))
        curr_prompt = f"Available actions are {ava_actions}, What are the steps needed to {task}."
    elif model_type == "ask_prompt":
        # prompt_for_prompt = "the steps to " + task[6:] + "in household tasks"
        prompt_for_prompt = task + " step by step"
        if args.language_model_type == "gpt3":
            response = openai.Completion.create(engine=planning_lm_id, prompt=prompt_for_prompt, **prompt_sampling_params)
            samples = [response['choices'][i]['text'] for i in range(prompt_sampling_params['n'])]
            
            log_probs = [np.mean(response['choices'][i]['logprobs']['token_logprobs']) for i in range(prompt_sampling_params['n'])]
        else:
            samples, log_probs = generator(prompt_for_prompt, prompt_sampling_params) 
        best_overall_score = -np.inf
        best_action = ""
        for sample, log_prob in zip(samples, log_probs):
            if args.open_loop:
                look_forward_program = sample
                ask_intent_prompt = look_forward_program + " Can you summarize my intent in 10 words?"
                if args.language_model_type == "gpt3":
                    # time.sleep(10)
                    response = openai.Completion.create(engine=planning_lm_id, prompt=ask_intent_prompt, **prompt_sampling_params)
                    samples_intent = [response['choices'][i]['text'] for i in range(prompt_sampling_params['n'])] 
                    log_probs_intent = [np.mean(response['choices'][i]['logprobs']['token_logprobs']) for i in range(prompt_sampling_params['n'])]
                else:
                    samples_intent, log_probs_intent = generator(ask_intent_prompt, prompt_sampling_params) 
                # best_intent_overall_score = -np.inf
                best_intent = samples_intent[0].strip().replace('\n', ' ')
                query_embedding = translation_lm.encode(task, convert_to_tensor=True, device=device)
                value_embedding = translation_lm.encode(best_intent, convert_to_tensor=True, device=device)
                intent_score = calc_textemb_distance(query_embedding, value_embedding)
                ic(best_intent, intent_score)
                overall_score = intent_score
            else:
                overall_score = BETA * log_prob
            if overall_score > best_overall_score:
                best_overall_score = overall_score
                best_action = sample
        if args.split_ask_prompt:
            translated_ask_prompt = ""
            curr_prompt_list = best_action.strip().split('\n')
            for ask_item in curr_prompt_list:
                translated_ask_idx, score = find_most_similar(ask_item, action_list_embedding)
                # translated_ask_idx, score = find_most_similar(ask_item+' '+task[6:], action_list_embedding)
                translated_ask_item = action_list[translated_ask_idx]
                if score < 0.5: continue
                translated_ask_prompt += translated_ask_item + '.'
                # ic(translated_ask_item, score)
            # ic(translated_ask_prompt)
            curr_prompt = translated_ask_prompt
        else:
            curr_prompt = best_action.strip().replace('\n', ' ')
        ic(task, curr_prompt)
    elif model_type == "concept_knowledge" or model_type == "cpall_learning_prompt":
        prompt_sample_num = args.intent_trial_num if args.summarize_intent else 1 
        for idx in range(prompt_sample_num):
            curr_prompt.append(get_concept_knowledge(task, idx*0.1))
    elif model_type == "chain_of_thought" or model_type == "chain_of_cause":
        exemplar_idx, _ = find_most_similar(task, chain_of_thought_task_embedding)
        exemplar = chain_of_thought_exemplar_list[exemplar_idx]
        example = chain_of_thought_program_list[exemplar_idx].replace('##STEP##', '\n')
        # ic(exemplar_idx, exemplar, example)
        curr_prompt.append(f'{exemplar}{example}\n\n{task}.')
    elif model_type == "task_only_base":
        curr_prompt.append(task+'.')
    else:       
        example_idx, _ = find_most_similar(task, example_task_embedding)
        example = heldout_available_examples[example_idx]
        # ic(len(example_task_embedding), example)
        curr_prompt.append(f'{example}\n\n{task}.')
    return curr_prompt

def translate_action(task, sample_item):
    # most_similar_idx, matching_score = find_most_similar(task + sample_item, action_list_embedding)
    most_similar_idx, matching_score = find_most_similar(sample_item, action_list_embedding)
    translated_action = action_list[most_similar_idx]
    return translated_action, matching_score

def language_planning(resultfile, total_score_cal, data_example, model_type, epoch_i=0):
    global example_task_embedding
    global action_list_embedding
    global result_list
    global generator

    task = data_example["tasks"]

    curr_prompt = get_current_prompt(model_type, task)
    # result_list.append('\n' + '-'*10 + ' GIVEN EXAMPLE ' + '-'*10+'\n')
    resultfile.write('\n' + '-'*10 + ' GIVEN EXAMPLE ' + '-'*10+'\n')
    task_eval_groundtruth = task + '. ' + str(data_example["steps"])
    # result_list.append('Task : '+task_eval_groundtruth+'\n')
    resultfile.write('Task : '+task_eval_groundtruth+'\n')

    task_eval_predict = ""
    task_prediced_steps = []

    if args.open_loop:
        if args.model_type == "zs_planner" or args.model_type == "concept_knowledge":
            # no prompt selection
            curr_prompt = curr_prompt[0]
            for generator_item in [generator]:
                samples, logbs = generator_item(curr_prompt + f' Step 1: ', sampling_params)
                best_sample, best_logb = "", -np.inf
                for sample, logb in zip(samples, logbs):
                    if logb > best_logb:
                        best_logb = logb
                        best_sample = sample
                task_eval_predict = ""
                best_sample = best_sample.lower()
                step_idx = 1
                for sample_idx, sample_item in enumerate(best_sample.split('step')):
                    sample_item = sample_item.replace('\n', " ")
                    # step-level translation
                    translated_action, matching_score = translate_action(task, sample_item)
                    matching_score += BETA * best_logb
                    if translated_action in task_eval_predict: matching_score -= 0.3
                    if matching_score < args.cut_threshold: continue
                    task_eval_predict += f" Step {step_idx}: " + translated_action + " ."
                    step_idx += 1
            resultfile.write('Task : '+ task + ". " + task_eval_predict + '\n')
        elif args.model_type == "intent_action_prompt" or args.model_type == "reward_action_prompt" or args.model_type == "intent_learning_prompt" or args.model_type == "all_learning_prompt" or args.model_type == "cpall_learning_prompt":
            # hitl_cycle_score
            # curr_prompt is a list of sampled prompts
            import time
            start = time.time()
            best_prompt_item, best_prompt_intent_score = "", -np.inf
            best_eval_predict = ""
            for curr_prompt_item in curr_prompt:
                for generator_item in [generator]:
                    samples, logbs = generator_item(curr_prompt_item + f' Step 1: ', sampling_params)
                    best_sample = ""
                    for sample, logb in zip(samples, logbs):
                        task_eval_predict = ""
                        sample = sample.lower()
                        step_idx = 1
                        for sample_idx, sample_item in enumerate(sample.split('\n')):
                            sample_item = sample_item.replace('\n', " ").replace(". ", ": ")
                            if len(sample_item) <= 8: continue
                            translated_step, matching_score = translate_action(task, sample_item)
                            # step-level intent
                            # paper exp table
                            # matching_score += (hitl_cycle_score(task, sample_item) * 0.25 - 0.26) if args.data_type == "robothow" else (hitl_cycle_score(task, sample_item) * 0.25 - 0.26)
                            # optimize
                            matching_score += (hitl_cycle_score(task, sample_item) * 0.25 - 0.26) if args.data_type == "robothow" else (hitl_cycle_score(task, sample_item) * 0.25 - 0.26)
                            if translated_step in task_eval_predict: matching_score -= 0.3
                            if matching_score < args.cut_threshold: continue
                            ic(sample_item, translated_step, matching_score)
                            task_eval_predict += f" Step {step_idx}: " + translated_step + " ."
                            step_idx += 1
                        # porgram-level selection
                        if args.data_type == "robothow":
                            prompt_intent_score = hitl_cycle_score(task, task_eval_predict) + BETA * logb
                        else:
                            prompt_intent_score = hitl_cycle_score(task, task_eval_predict) * 0.05 + BETA * logb
                            ic(best_prompt_intent_score, task_eval_predict, prompt_intent_score)
                        if prompt_intent_score > best_prompt_intent_score:
                            best_prompt_intent_score = prompt_intent_score
                            best_prompt_item = curr_prompt_item
                            best_eval_predict = task_eval_predict
            # ic(time.time()-start, best_prompt_item, task_eval_predict)

            # for generator_item in [generator]:
            #     sample = generator_item(curr_prompt, sampling_params)[0][0]
            #     task_eval_predict = ""
            #     first_step = sample.split('\n')[0]
            #     task_eval_predict += "Step 1: " + translate_action(task, first_step) + " "
            #     for sample_idx, sample_item in enumerate(sample.split('\n')[1:]):
            #         sample_item = sample_item.replace('\n', " ").replace(". ", ": ")
            #         task_eval_predict += f". Step {sample_idx+2}: " + translate_action(task, sample_item) + " "
            resultfile.write('Task : '+ task + ". " + best_eval_predict + '\n')
        elif args.model_type == "action_prompt":
            for generator_item in [generator]:
                sample = generator_item(curr_prompt, sampling_params)[0][0]
                task_eval_predict = ""
                sample = sample.lower()
                step_idx = 1
                for sample_idx, sample_item in enumerate(sample.split('\n')):
                    translated_step, matching_score = translate_action(task, sample_item.replace('\n', " ").replace(". ", ": "))
                    if translated_step in task_eval_predict or matching_score < args.cut_threshold: continue
                    task_eval_predict += f" Step {step_idx}: " + translated_step + " ."
            resultfile.write('Task : '+ task + ". " + task_eval_predict + '\n')
        elif args.model_type == "ask_prompt":
            task_prediced_steps = curr_prompt.split('.')
            task_eval_predict = ""
            for step_i, step_item in enumerate(task_prediced_steps):
                if len(step_item) <= 0: continue
                task_eval_predict += f" Step {step_i + 1}: {step_item} ."
            # result_list.append('Task : '+task + ". " + task_eval_predict + '\n')
            resultfile.write('Task : '+task + ". " + task_eval_predict + '\n')
    else:
        for generator_item in [generator]:
            # result_list.append(f'Task : {task}.')
            resultfile.write(f'Task : {task}.')
            step_sequence = []
            for step in range(1, MAX_STEPS + 1):
                if args.language_model_type == "bart":
                    sample = generator_item(curr_prompt + f' Step {step}: ', sampling_params)
                    most_similar_idx, matching_score = find_most_similar(sample, action_list_embedding)
                    translated_action = action_list[most_similar_idx]
                    if translated_action in task_prediced_steps: matching_score -=0.3
                    if matching_score < args.cut_threshold: break
                    best_action = translated_action
                    previous_action = best_action
                    task_prediced_steps.append(best_action)
                    formatted_action = (best_action[0].upper() + best_action[1:]).replace('_', ' ')
                    curr_prompt += f'\nStep {step}: {formatted_action}.'
                    # result_list.append(f' Step {step}: {formatted_action}.')
                    resultfile.write(f' Step {step}: {formatted_action}.')
                    step_sequence.append(best_action)
                    task_eval_predict += f' Step {step}: {formatted_action}.'
                else:
                    best_overall_score = -np.inf
                    curr_prompt = curr_prompt.replace("\n", " ")
                    samples, log_probs = generator_item(curr_prompt + f' Step {step}: ', sampling_params) 
                    for sample, log_prob in zip(samples, log_probs):
                        most_similar_idx, matching_score = find_most_similar(sample, action_list_embedding)
                        overall_score = matching_score + BETA * log_prob
                        translated_action = action_list[most_similar_idx]
                        if translated_action in task_prediced_steps:
                            overall_score -= 0.5
                        if step > 1 and translated_action == previous_action:
                            overall_score -= 0.5
                        # summarize_intent score
                        if args.summarize_intent:
                            look_forward_program = task_eval_predict + f' Step {step}: {translated_action}.'
                            ask_intent_prompt = look_forward_program + " Can you summarize my intent in 10 words?"
                            if args.language_model_type == "gpt3":
                                # time.sleep(10)
                                response = openai.Completion.create(engine=planning_lm_id, prompt=ask_intent_prompt, **prompt_sampling_params)
                                samples_intent = [response['choices'][i]['text'] for i in range(prompt_sampling_params['n'])] 
                                log_probs_intent = [np.mean(response['choices'][i]['logprobs']['token_logprobs']) for i in range(prompt_sampling_params['n'])]
                            else:
                                samples_intent, log_probs_intent = generator(ask_intent_prompt, prompt_sampling_params) 
                            # best_intent_overall_score = -np.inf
                            best_intent = samples_intent[0].strip().replace('\n', ' ')
                            # for sample_intent, log_prob_intent in zip(samples_intent, log_probs_intent):
                            #     sample_intent = sample_intent.strip()
                            #     overall_intent_score = BETA * log_prob_intent
                            #     if overall_intent_score > best_intent_overall_score:
                            #         best_intent_overall_score = overall_score
                            #         best_intent = sample_intent
                            query_embedding = translation_lm.encode(task, convert_to_tensor=True, device=device)
                            value_embedding = translation_lm.encode(best_intent, convert_to_tensor=True, device=device)
                            intent_score = calc_textemb_distance(query_embedding, value_embedding)
                            best_overall_score += intent_score
                            ic(task, best_intent, intent_score)
                        if overall_score > best_overall_score:
                            best_overall_score = overall_score
                            best_action = translated_action
                    ic(overall_score, best_action)
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
                        # result_list.append(f' Step {step}: {formatted_action}.')
                        resultfile.write(f' Step {step}: {formatted_action}.')
                        step_sequence.append(best_action)
                        task_eval_predict += f' Step {step}: {formatted_action}.'
            if args.summarize_intent:
                # result_list.append('\n' + '-'*10 + ' BEST INTENT ' + '-'*10 + best_intent + '\n')
                resultfile.write('\n' + '-'*10 + ' BEST INTENT ' + '-'*10 + best_intent + '\n')

    # return calculate_total_score(total_score_cal, model_type, task_eval_groundtruth, task + ". " + task_eval_predict)
    if args.use_intent_score:
        look_forward_program = task_eval_predict
        ask_intent_prompt = look_forward_program + " Can you summarize my intent in 10 words?"
        if args.language_model_type == "gpt3":
            response = openai.Completion.create(engine=planning_lm_id, prompt=ask_intent_prompt, **prompt_sampling_params)
            samples_intent = [response['choices'][i]['text'] for i in range(prompt_sampling_params['n'])] 
            log_probs_intent = [np.mean(response['choices'][i]['logprobs']['token_logprobs']) for i in range(prompt_sampling_params['n'])]
        else:
            samples_intent, log_probs_intent = generator(ask_intent_prompt, prompt_sampling_params) 
        best_intent = samples_intent[0].strip().replace('\n', ' ')
        query_embedding = translation_lm.encode(task, convert_to_tensor=True, device=device)
        value_embedding = translation_lm.encode(best_intent, convert_to_tensor=True, device=device)
        metric_intent_score = [calc_textemb_distance(query_embedding, value_embedding)]
    else:
        metric_intent_score = [0]

    cur_metric_line = get_metric_csv_line(model_type, task_eval_groundtruth, task + ". " + task_eval_predict, metric_intent_score)
    metric_csv_lines.append(cur_metric_line)
    return cur_metric_line


if args.model_type == "chain_of_thought" or args.model_type == "chain_of_cause":
    f = open("../data/{}/{}_chain_of_thought.json".format(args.data_type, args.data_type))
    data = json.load(f)
    category_exemplars = data["exemplars"]["category_foodpreparation"] if args.data_type == "robothow" else data["exemplars"]["category_other"]
    chain_of_thought_task_list = [exemplar["task"] for exemplar in category_exemplars]
    chain_of_thought_exemplar_list = [exemplar["program_chain"] for exemplar in category_exemplars]

    if args.model_type == "chain_of_thought":
        chain_of_thought_program_list = [exemplar["program_chain"] + "##PROGRAM##" + "##STEP##".join(exemplar["program"].split('\n')) for exemplar in category_exemplars]
    else:
        chain_of_thought_program_list = [exemplar["program_chain"] + exemplar["program_causal_chain"] + "##PROGRAM##" + "##STEP##".join(exemplar["program"].split('\n')) for exemplar in category_exemplars]

total_score_cal = total_score.copy()
task_result_dir = os.path.join("../result", task_path)
if not os.path.isdir(task_result_dir): os.makedirs(task_result_dir)
skip_count = 0
metric_csv_lines = []
with open(os.path.join(task_result_dir, "{}_sum{}_intent{}_task_result_metric.csv".format(args.language_model_type, args.objective_summarization, args.summarize_intent)), 'w') as csvfile:
    writer = csv.writer(csvfile, delimiter=',', quoting=csv.QUOTE_MINIMAL)
    head_line = ['method_type', 's_bleu', 'sim', 'bert_f1', 'rouge-1-f1', 'bert_f1_norm', 'rouge_l_f1', 'mover', 'intent', 'avg_length']
    writer.writerow(head_line)
    with open(os.path.join(task_result_dir, 'commandline_args.txt'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)
    with open(os.path.join(task_result_dir, "{}_sum{}_intent{}_task_result.txt".format(args.language_model_type, args.objective_summarization, args.summarize_intent)), 'w') as resultfile:
        total_score_cal = total_score.copy()
        for data_idx, data_example in enumerate(summarize_example_data_list):
            metric_csv_line = language_planning(resultfile, total_score_cal, data_example, args.model_type)
            writer.writerow(metric_csv_line)


    # ic(len(summarize_example_data_list), total_score_cal[args.model_type].keys())
    # for score_key in total_score_cal[args.model_type].keys():
    #     total_score_cal[args.model_type][score_key] /= (len(summarize_example_data_list)-skip_count)
    # json.dump(total_score_cal,resultfile)
    # ic(skip_count, total_score_cal[args.model_type])



    # for metric_csv_line in metric_csv_lines:
    #     writer.writerow(metric_csv_line)