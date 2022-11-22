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
positiveStrategy,simulator, records, total_regret = pickle.load(open('surprise_bandit_epoch_0_11.11','rb'))
txts = ["Tom is eating a hotdog on a train."]
crt_lenght = 1
b_factors = [10,30,60]


def test_mode(mode,prompt):
    prompts_ppls = [[prompt,1]]
    crt_length = 0
    crt_budget = 10000
    while True:
        crt_length = crt_length + 1
        (txts_at_timestep, overall_sample_features, overall_rewards, best_per_bucket, buckets) = dg.generate_gpt_topk_example(prompts_ppls,crt_length, crt_budget)

        # print(overall_sample_features)
        l = [positiveStrategy.estimate(0, overall_sample_features), positiveStrategy.estimate(1, overall_sample_features), \
         positiveStrategy.estimate(2, overall_sample_features)]
        ll = [positiveStrategy.estimate(0, overall_sample_features[0]), positiveStrategy.estimate(1, overall_sample_features[0]), \
         positiveStrategy.estimate(2, overall_sample_features[0])]
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
            arm = np.argmax(ll)
        # arm = random.choice([0,1,2])
        # regret, rmse, arm = simulator.simulate(overall_sample_features, overall_rewards, dg.W)
        print("choice of arm is: ",arm )

        # tmps,_ = topk_surprise_sampling_one(model,tokenizer,txts,b_factors[arm])
        # txts = tmps
        chosen = buckets[arm]
        chosen.sort(key=lambda tup: tup[1], reverse=True)


        chosen = chosen[:b_factors[arm]]
        prompts_ppls = [(i[0], i[-1]) for i in chosen]

        if crt_length >8:
            break

    rankee = []
    already = []
    txts = [i[0] for i in prompts_ppls]
    for txt in txts:
        # v = get_valence_score(txt.split(" "))[0]

        words = txt.split(" ")
        a = get_valence_score(words)[0]

    return mode, txts[0],a
dumpee = []
modes = ["small","medium","large","random","planned"]
from tqdm import tqdm
for context in tqdm(contexts[10000:10100]):
    savee = []
    for mode in modes:
        rt = test_mode(mode,context.strip(" "))
        savee.append(rt)
    savee.sort(key=lambda tup: tup[2], reverse=True)
    dumpee.append(savee)
print(dumpee)
import pickle
pickle.dump(dumpee, open("evaluation.p","wb"))

# not 0 is correct
