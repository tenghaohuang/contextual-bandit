import argparse
import csv
from nltk.stem.wordnet import WordNetLemmatizer
from collections import defaultdict, Counter
import pandas as pd
from scipy.stats import zscore
import math
import warnings
# from pandas.core.common import SettingWithCopyWarning
import pickle
from utils import *
import scipy.stats as ss
from heapq import nlargest
from config import Config

def get_NRC_lexicon(path):
    '''
    @output:
    - A dictionary of format {word : score}
    '''
    lexicon = path
    val_dict = {}
    aro_dict = {}
    dom_dict = {}
    with open(lexicon, 'r') as infile:
        reader = csv.DictReader(infile, delimiter="\t")
        for row in reader:
            word = row['Word']
            val_dict[word] = float(row['Valence'])
            aro_dict[word] = float(row['Arousal'])
            dom_dict[word] = float(row['Dominance'])
    return (val_dict, aro_dict, dom_dict)


val_dict, aro_dict, _ = get_NRC_lexicon("/nas/luka-group/tenghao/tenghaoh/creative_writing/Finetuned_GPT2/dataset/NRC-VAD-Lexicon.txt")

def get_arousal_score(infs):
    '''
    input:
        infs: a list of commonsense inferences
    output:
        score: the sum of valence valence scores
    '''
    if infs ==[]:
        return None,None
    sum = 0
    # print(infs)
    rt_l = []
    for inf in infs:
        inf = remove_stop_words(inf)

        sub_scores = []
        cnt = 0
        for part in inf:
            if part not in aro_dict:
                continue
            cnt+=1
            sub_scores.append(aro_dict[part])
        if sub_scores == []:
            continue
        sum+= max(sub_scores)
        rt_l.append(max(sub_scores))
    return sum,rt_l

def get_valence_score(infs):
    '''
    input:
        infs: a list of commonsense inferences
    output:
        score: the sum of valence valence scores
    '''
    if infs ==[]:
        return None,None
    sum = 0
    # print(infs)
    rt_l = []
    for inf in infs:
        inf = remove_stop_words(inf)

        sub_scores = []
        cnt = 0
        for part in inf:
            if part not in val_dict:
                continue
            cnt+=1
            sub_scores.append(val_dict[part])
        if sub_scores == []:
            continue
        sum+= max(sub_scores)
        rt_l.append(max(sub_scores))
    return sum,rt_l

def get_most_opp_valence(args, first_stage, second_stage,model,tokenizer):
    most_oppo_val_stories = []
    for setting_gene, cont_genes in zip(first_stage, chunks(second_stage, args.rounds_of_generation)):
        setting_inf_score = get_valence_score(setting_gene[args.surprise_position - 1]['<|xReact|>'])
        cont_gap_scores = []
        for cont_gene in cont_genes:
            # print(cont_gene)
            cont_inf_score = get_valence_score(cont_gene[args.surprise_position]['<|xReact|>'])
            cont_gap_scores.append(abs(cont_inf_score - setting_inf_score))
        prompt = cont_genes[cont_gap_scores.index(max(cont_gap_scores))]['story']
        _, full_story = finish_story(prompt, model, tokenizer, num_story_return=1)
        most_oppo_val_stories.append(full_story)
    return most_oppo_val_stories

def get_surprising_generations(input_path, output_path):
    GPT2_sampling_mode = get_GPT2_sampling_mode(input_path)
    if GPT2_sampling_mode == "normal":
        OUTPUT_NAME = output_path + "normal_contract_action_"
    elif GPT2_sampling_mode =="paracomet":
        OUTPUT_NAME = output_path + "paracomet_contract_action_"
    else:
        assert(False)
    gene_inferences = getCommonsenseOutputs(input_path)
    setting_inf_score = get_valence_score(gene_inferences[0][1]['<|xReact|>'])
    cont_inf_scores = []
    for all_inf in gene_inferences:
        cont_inf_score = get_valence_score(all_inf[2]['<|xReact|>'])
        cont_inf_scores.append(cont_inf_score)
    cont_inf_scores = [abs(i - setting_inf_score) for i in cont_inf_scores]
    ranks = list(ss.rankdata(cont_inf_scores,method='ordinal'))
    ranks = [int(i) for i in ranks]
    top_five = [i for i in range(len(ranks)-5,len(ranks))]
    selected_ids = [ranks.index(q) for q in top_five]
    selected_genes = [gene_inferences[id] for id in selected_ids]
    saved_name = input_path.split("/")[-1].split('_')[0]
    f = open(OUTPUT_NAME + saved_name+".txt", "w")
    gene_stories = []
    for gene in selected_genes:
        f.write(str(gene[1]['<|xReact|>'])+"\n")
        f.write(str(gene[2]['<|xReact|>'])+"\n")
        f.write(gene['story']+"\n")
        gene_stories.append(gene['story'])
    f.write("\n")
    f.write("*******************************************\n")

    f.write("++++++++++++++++++++++++++++++++++++++++++++\n")
    for gene in gene_inferences:
        f.write(gene['story']+"\n")
    f.close()
    return gene_stories
# in_dir = "/playpen-ssd/tenghao/creative_writing/Finetuned_GPT2/commonsense_output/anti_paracomet_prob/"
#
# get_surprising_generations(in_dir+ "0_inferences.jsonl",Config.selector_output_path)
#
