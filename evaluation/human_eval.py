import pandas as pd
from icecream import ic
import argparse
import csv

parser = argparse.ArgumentParser(description='Intervention')
parser.add_argument('--load_result_path', type=str, default="", help='load result filepath')
parser.add_argument('--data_type', choices=['wikihow', 'robothow', 'wiki_inter', 'robot_inter'], default='robothow', help='choices')
parser.add_argument('--sample_for_eval', type=int, default=50)
parser.add_argument('--sample_per_worker', type=int, default=7)
parser.add_argument('--model_type', choices=['human', 'concept_knowledge', 'task_only_base', 'base', 'base_tune', 'standard_prompt', 'soft_prompt_tuning', 'chain_of_thought', 'chain_of_cause', 'cmle_ipm', 'cmle_epm', 'irm', 'vae_r', 'rwSAM', 'counterfactual_prompt'], default='base', help='choices')
args = parser.parse_args()

path_set = [
    "/Users/USERNAME/Desktop/{}/chain-gpt2-xl_sumFalse_task_result.txt".format(args.data_type),
    "/Users/USERNAME/Desktop/{}/planner-gpt2-xl_sumFalse_task_result.txt".format(args.data_type),
    "/Users/USERNAME/Desktop/{}/concept-gpt2-xl_sumFalse_task_result.txt".format(args.data_type),
    "/Users/USERNAME/Desktop/{}/chain-bart_sumFalse_task_result.txt".format(args.data_type),
    "/Users/USERNAME/Desktop/{}/planner-bart_sumFalse_task_result.txt".format(args.data_type),
    "/Users/USERNAME/Desktop/{}/concept-bart_sumFalse_task_result.txt".format(args.data_type),
]

path_set_to_model = {
    "/Users/USERNAME/Desktop/{}/chain-gpt2-xl_sumFalse_task_result.txt".format(args.data_type): "gpt_chain",
    "/Users/USERNAME/Desktop/{}/planner-gpt2-xl_sumFalse_task_result.txt".format(args.data_type): "gpt_planner",
    "/Users/USERNAME/Desktop/{}/concept-gpt2-xl_sumFalse_task_result.txt".format(args.data_type): "gpt_concept",
    "/Users/USERNAME/Desktop/{}/chain-bart_sumFalse_task_result.txt".format(args.data_type): "bart_chain",
    "/Users/USERNAME/Desktop/{}/planner-bart_sumFalse_task_result.txt".format(args.data_type): "bart_planner",
    "/Users/USERNAME/Desktop/{}/concept-bart_sumFalse_task_result.txt".format(args.data_type): "bart_concept",
}
with open("{}_humaneval.csv".format(args.data_type), 'w') as csvfile:
    writer = csv.writer(csvfile, delimiter=',', quoting=csv.QUOTE_MINIMAL)
    head_line = ['example_idx']
    ic(args.sample_per_worker)
    for idx in range(args.sample_per_worker):
        head_line += ['program{}_idx'.format(idx+1)] + ['task{}_txt'.format(idx+1)] + ['steps{}_txt'.format(idx+1)]
    writer.writerow(head_line)

    program_idx = 0
    sample = 0    
    is_human = True
    program_list_map = {"human":[], "gpt_chain":[], "gpt_planner":[], "gpt_concept":[], "bart_chain":[], "bart_planner":[], "bart_concept":[]}
    for file_path in path_set:
        ic(file_path)
        sample = 0
        with open(file_path, 'r') as fp:
            # head line
            line = fp.readline()
            while line and sample < args.sample_for_eval: 
                if "GIVEN EXAMPLE" in line:
                    sample += 1
                    gt_line = fp.readline()
                    if is_human:
                        program_list_map["human"].append(gt_line.strip())   
                    line = fp.readline()
                    program_list_map[path_set_to_model[file_path]].append(line.strip())
                line = fp.readline()
        is_human = False


    csv_line = []
    ic(len(program_list_map["human"]))
    for example_idx in range(len(program_list_map["human"])):
        program_idx += 1
        csv_line += [str(example_idx+1)] + [str(program_idx)] + [program_list_map["human"][example_idx].split('.')[0]] + ['<br>'.join(program_list_map["human"][example_idx].split('.')[1:])]
        program_idx += 1
        csv_line += [str(program_idx)] + [program_list_map["gpt_chain"][example_idx].split('.')[0]] + ['<br>'.join(program_list_map["gpt_chain"][example_idx].split('.')[1:])]
        program_idx += 1
        csv_line += [str(program_idx)] + [program_list_map["gpt_planner"][example_idx].split('.')[0]] + ['<br>'.join(program_list_map["gpt_planner"][example_idx].split('.')[1:])]
        program_idx += 1
        csv_line += [str(program_idx)] + [program_list_map["gpt_concept"][example_idx].split('.')[0]] + ['<br>'.join(program_list_map["gpt_concept"][example_idx].split('.')[1:])]
        program_idx += 1
        csv_line += [str(program_idx)] + [program_list_map["bart_chain"][example_idx].split('.')[0]] + ['<br>'.join(program_list_map["bart_chain"][example_idx].split('.')[1:])]
        program_idx += 1
        csv_line += [str(program_idx)] + [program_list_map["bart_planner"][example_idx].split('.')[0]] + ['<br>'.join(program_list_map["bart_planner"][example_idx].split('.')[1:])]
        program_idx += 1
        csv_line += [str(program_idx)] + [program_list_map["bart_concept"][example_idx].split('.')[0]] + ['<br>'.join(program_list_map["bart_concept"][example_idx].split('.')[1:])]
        writer.writerow(csv_line)
        csv_line = []