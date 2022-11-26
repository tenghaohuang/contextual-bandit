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
from utils import wrapup_input
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
positiveStrategy,simulator, records, total_regret = pickle.load(open("/playpen-ssd/tenghao/creative_writing/surprise_bandit/surprise_bandit_epoch_e_reward_epoch1",'rb'))
txts = ["Tom is eating a hotdog on a train."]

crt_lenght = 1
b_factors = [10,30,60]


def test_mode(mode,prompt, stop_at):
    prompts_ppls = [[prompt,1]]
    crt_length = 0
    crt_budget = 10000
    cnt = 0
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
        chosen.sort(key=lambda tup: tup[1], reverse=True)


        chosen = chosen[:b_factors[arm]]
        prompts_ppls = [(i[0], i[-1]) for i in chosen]

        if crt_length >stop_at:
            break

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
for context in tqdm(contexts[10000:10100]):
    if context.endswith(" "):
        prompts_ppls = [[wrapup_input(context[:-1], 'prompt'), 1]]
    else:
        prompts_ppls = [[wrapup_input(context, 'prompt'), 1]]
    savee = []
    for mode in modes:
        mode, txts[0],a, cnt = test_mode(mode,context.strip(" "), 4)
        rt = (mode, txts[0],a)
        savee.append(rt)
        total_cnt += cnt
        print(rt[1])
    savee.sort(key=lambda tup: tup[2], reverse=True)
    dumpee.append(savee)
print(total_cnt)

# print(dumpee)
# import pickle
# pickle.dump(dumpee, open("evaluation_4_planned_gold.p","wb"))
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
