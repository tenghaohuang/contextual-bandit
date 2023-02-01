import math
import random

import DataGenerator
from utils import getStories, get_transformer
from config import Config
# import Simulator
from IPython import embed
from valence_measure import get_valence_score, get_arousal_score
import pickle
import numpy as np
import DataGenerator
from utils import wrapup_input, finish_story
from proof_of_concept_experiment import topk_surprise_sampling_one
from sampling_during_decoding import count_punct
arms=3
features=4
rewardType='positive'
#rewardType='binary'
featureType='integer'
#featureType='binary'

# define number of samples and number of choices
contexts, references = getStories(Config.story_path, story_num=50000, surprise_position=1)
dg = DataGenerator.DataGenerator(arms, features, feature_type=featureType, reward_type=rewardType)

# model, tokenizer = get_transformer(Config.GPT2_finetuned_ROC)
# model.cuda()
# model.eval()
# open('surprise_bandit_epoch_0','rb').close()
positiveStrategy,simulator, records, total_regret = pickle.load(open("/nas/luka-group/tenghao/tenghaoh/creative_writing/surprise_bandit/surprise_bandit_epoch_1",'rb'))
txts = ["Tom is eating a hotdog on a train."]

crt_lenght = 1
b_factors = [10,30,60]

puncts = [".","!","?"]
def check_puncts(txt):
    cnt = 0
    clean = ""
    for s in txt:
        if s in puncts:
            cnt +=1
        else:
            clean+=s
    return cnt,clean
def length_penalty(gama, alpha=1):
    return (5+gama)**alpha/6**alpha
def test_mode(mode,prompt, stop_at):
    prompts_ppls = [[prompt,1]]
    crt_length = 0
    crt_budget = 10000
    cnt = 0
    finished = []
    minimum_len = 4
    collection = []
    while True:
        crt_length = crt_length + 1
        (txts_at_timestep, overall_sample_features, overall_rewards, best_per_bucket, buckets) = dg.generate_gpt_topk_example(prompts_ppls,crt_length, crt_budget)

        # print(overall_sample_features)

        # print(l)
        if mode == "small":
            arm = 0
        elif mode == "medium":
            arm =1
        elif mode == "large":
            arm = 2
        elif mode == "random":
            arm = random.choice([0,1,2])
        elif mode == "planned":
            l = [positiveStrategy.estimate(0, overall_sample_features),
                 positiveStrategy.estimate(1, overall_sample_features), \
                 positiveStrategy.estimate(2, overall_sample_features)]
            ll = [positiveStrategy.estimate(0, overall_sample_features[0]),
                  positiveStrategy.estimate(1, overall_sample_features[0]), \
                  positiveStrategy.estimate(2, overall_sample_features[0])]
            arm = np.argmax(ll)
            gold = np.argmax(overall_rewards)
            if arm==gold:
                cnt += 1
        # arm = random.choice([0,1,2])
        # regret, rmse, arm = simulator.simulate(overall_sample_features, overall_rewards, dg.W)
        #TODO: now we are doing gold choices
        print("choice of gold arm is: ",gold )

        # tmps,_ = topk_surprise_sampling_one(model,tokenizer,txts,b_factors[arm])
        # txts = tmps
        chosen = buckets[arm]
        cap_num = 10
        new_chosen = []
        for tup in chosen:
            punct_num, clean_txt = check_puncts(tup[0])
            tup = list(tup)
            tup.append(crt_length)
            if crt_length>minimum_len and punct_num==2:
                if clean_txt not in collection and crt_length>minimum_len and cap_num >0:
                    finished.append(tup)
                    collection.append(clean_txt)
                    cap_num-=1
                tup[1]+=1
            else:
                new_chosen.append(tup)
        chosen = new_chosen
        chosen.sort(key=lambda tup: tup[1], reverse=True)


        chosen = chosen[:b_factors[arm]]
        prompts_ppls = [(i[0], i[-1]) for i in chosen]

        # if crt_length >stop_at:
        #     break
        # if crt_length>minimum_len and crt_length==10:
        #     embed()
        if crt_length>minimum_len and len(finished)>50:
            # embed()
            break

    prompts_ppls = [(i[0], i[1]/length_penalty(i[-1])) for i in finished]
    prompts_ppls.sort(key=lambda tup: tup[1], reverse=True)
    rankee = []
    already = []
    txts = [i[0] for i in prompts_ppls]

        # v = get_valence_score(txt.split(" "))[0]

    words = txts[0].split(" ")
    a = get_arousal_score(words)[0]

    return mode, txts[0],a, cnt

modes = ["planned","small","medium","large","random"]
modes = ['planned']
from tqdm import tqdm
dumpee = []
total_cnt = 0
def get_ascore(txt):
    words = txt.split(" ")
    a = get_arousal_score(words)[0]/len(words)
    return a

for context in tqdm(contexts[10000:10010]):
    if context.endswith(" "):
        prompts_ppls = [[wrapup_input(context[:-1], 'prompt'), 1]]
    else:
        prompts_ppls = [[wrapup_input(context, 'prompt'), 1]]
    savee = []
    for mode in modes:
        mode, txts[0],a, cnt = test_mode(mode,context.strip(" "), 10)
        rt = (mode, txts[0],a)

        total_cnt += cnt
        finished = finish_story(rt[1], dg.model, dg.tokenizer, beam_size=10,num_story_return=1)
        finished = " ".join(finished[0])
        print(finished)
        savee.append([rt[0], finished, get_ascore(finished)])
    savee.sort(key=lambda tup: tup[2], reverse=True)
    dumpee.append(savee)

print(total_cnt)

# print(dumpee)
import pickle
pickle.dump(dumpee, open("evaluation_planned_mine.p","wb"))
#
# modes = ['planned']
# dumpee = []
# for context in tqdm(contexts[10000:10100]):
#     if context.endswith(" "):
#         prompts_ppls = [[wrapup_input(context[:-1], 'prompt'), 1]]
#     else:
#         prompts_ppls = [[wrapup_input(context, 'prompt'), 1]]
#     savee = []
#     for mode in modes:
#         rt = test_mode(mode,context.strip(" "), 6)
#         savee.append(rt)
#         print(rt[1])
#     savee.sort(key=lambda tup: tup[2], reverse=True)
#     dumpee.append(savee)
# print(dumpee)
# import pickle
# pickle.dump(dumpee, open("evaluation_6_planned_gold.p","wb"))
#
# modes = ['planned']
#
# dumpee = []
# for context in tqdm(contexts[10000:10100]):
#     if context.endswith(" "):
#         prompts_ppls = [[wrapup_input(context[:-1], 'prompt'), 1]]
#     else:
#         prompts_ppls = [[wrapup_input(context, 'prompt'), 1]]
#     savee = []
#     for mode in modes:
#         rt = test_mode(mode,context.strip(" "), 8)
#         savee.append(rt)
#         print(rt[1])
#     savee.sort(key=lambda tup: tup[2], reverse=True)
#     dumpee.append(savee)
# print(dumpee)
# import pickle
# pickle.dump(dumpee, open("evaluation_8_planned_gold.p","wb"))

# not 0 is correct
