import csv
import pandas as pd
import argparse
from icecream import ic

parser = argparse.ArgumentParser(description='Intervention')
parser.add_argument('--model_base_type', choices=['bart', 'gpt', 'gpt3'], default="gpt3", help='load result filepath')
# parser.add_argument('--metric_type', choices=['plan', 'order'], default="plan", help='load result filepath')
# parser.add_argument('--data_type', choices=['wikihow', 'robothow', 'wiki_inter', 'robot_inter'], default='robothow', help='choices')
args = parser.parse_args()
score_map = {}
for data_type in ['wikihow', 'robothow', 'wiki_inter', 'robot_inter']:
    if not data_type in score_map.keys():
        score_map[data_type] = {}    
        for metric_type in ['plan', 'order']:
            if not metric_type in score_map[data_type].keys() is None:
                score_map[data_type][metric_type] = {}
            for model_type in ['gpt3']:#['bart', 'gpt']:
                df = pd.read_csv("winlose_data/{}_{}_{}.csv".format(metric_type, data_type, model_type))
                # ic(df['Answer.semantic-similarity.label'])
                
                if metric_type == "plan":
                    df.loc[pd.isna(df["Input.steps1_txt"]), 'Answer.semantic-similarity.label'] = "3 - Sequence 2 covers more"
                    df.loc[pd.isna(df["Input.steps2_txt"]), 'Answer.semantic-similarity.label'] = "1 - Sequence 1 covers more"
                    ours = len(df.loc[df['Answer.semantic-similarity.label'] == "3 - Sequence 2 covers more"])
                    equal = len(df.loc[df['Answer.semantic-similarity.label'] == "2 - Tie"])
                    base = len(df.loc[df['Answer.semantic-similarity.label'] == "1 - Sequence 1 covers more"])
                else:
                    df.loc[pd.isna(df["Input.steps1_txt"]), 'Answer.semantic-similarity.label'] = "3 - Sequence 2 is better"
                    df.loc[pd.isna(df["Input.steps2_txt"]), 'Answer.semantic-similarity.label'] = "1 - Sequence 1 is better"
                    ours = len(df.loc[df['Answer.semantic-similarity.label'] == "3 - Sequence 2 is better"])
                    equal = len(df.loc[df['Answer.semantic-similarity.label'] == "2 - Tie"])
                    base = len(df.loc[df['Answer.semantic-similarity.label'] == "1 - Sequence 1 is better"])
                total = ours+equal+base
                score_map[data_type][metric_type]['{}_concept'.format(model_type)] = round(ours/total*100, 2)
                score_map[data_type][metric_type]['{}_planner'.format(model_type)] = round(base/total*100, 2)
                score_map[data_type][metric_type]['{}_equal'.format(model_type)] = round(equal/total*100, 2)
                print(data_type + ' ' + model_type + ' ' + metric_type, round(ours/total*100, 2), round(equal/total*100, 2), round(base/total*100, 2))
ic(score_map)
score_list = []
if args.model_base_type == "bart":
    for model_type in ['bart_concept', 'bart_equal', 'bart_planner']:
        score_list.append(str(score_map['robothow']['plan'][model_type]))

    for model_type in ['bart_concept', 'bart_equal', 'bart_planner']:
        score_list.append(str(score_map['robothow']['order'][model_type]))

    for model_type in ['bart_concept', 'bart_equal', 'bart_planner']:
        score_list.append(str(score_map['robot_inter']['plan'][model_type]))

    for model_type in ['bart_concept', 'bart_equal', 'bart_planner']:
        score_list.append(str(score_map['robot_inter']['order'][model_type]))


    print('&' + '&'.join(score_list) + '\\') 

    score_list = []
    for model_type in ['bart_concept', 'bart_equal', 'bart_planner']:
        score_list.append(str(score_map['wikihow']['plan'][model_type]))


    for model_type in ['bart_concept', 'bart_equal', 'bart_planner']:
        score_list.append(str(score_map['wikihow']['order'][model_type]))


    for model_type in ['bart_concept', 'bart_equal', 'bart_planner']:
        score_list.append(str(score_map['wiki_inter']['plan'][model_type]))


    for model_type in ['bart_concept', 'bart_equal', 'bart_planner']:
        score_list.append(str(score_map['wiki_inter']['order'][model_type]))

    print('&' + '&'.join(score_list) + '\\') 
elif args.model_base_type in ["gpt", "gpt3"]:
    for model_type in ['gpt3_concept', 'gpt3_equal', 'gpt3_planner']:
        score_list.append(str(score_map['robothow']['plan'][model_type]))

    for model_type in ['gpt3_concept', 'gpt3_equal', 'gpt3_planner']:
        score_list.append(str(score_map['robothow']['order'][model_type]))

    for model_type in ['gpt3_concept', 'gpt3_equal', 'gpt3_planner']:
        score_list.append(str(score_map['robot_inter']['plan'][model_type]))

    for model_type in ['gpt3_concept', 'gpt3_equal', 'gpt3_planner']:
        score_list.append(str(score_map['robot_inter']['order'][model_type]))


    print('&' + '&'.join(score_list) + '\\') 

    score_list = []
    for model_type in ['gpt3_concept', 'gpt3_equal', 'gpt3_planner']:
        score_list.append(str(score_map['wikihow']['plan'][model_type]))


    for model_type in ['gpt3_concept', 'gpt3_equal', 'gpt3_planner']:
        score_list.append(str(score_map['wikihow']['order'][model_type]))


    for model_type in ['gpt3_concept', 'gpt3_equal', 'gpt3_planner']:
        score_list.append(str(score_map['wiki_inter']['plan'][model_type]))


    for model_type in ['gpt3_concept', 'gpt3_equal', 'gpt3_planner']:
        score_list.append(str(score_map['wiki_inter']['order'][model_type]))

    print('&' + '&'.join(score_list) + '\\') 
