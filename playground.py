import math

import DataGenerator
from utils import getStories, get_transformer
from config import Config
# import Simulator
from IPython import embed
from valence_measure import get_valence_score, get_arousal_score
import pickle
import numpy as np
from proof_of_concept_experiment import topk_surprise_sampling_one
from sampling_during_decoding import count_punct
model, tokenizer = get_transformer(Config.GPT2_finetuned_ROC)
model.cuda()
model.eval()
positiveStrategy,simulator, records, total_regret = pickle.load(open('surprise_bandit_epoch_4','rb'))
txts = ["Tom is eating a hotdog on a train."]
crt_lenght = 1
b_factors = [10,30,60]


while True:
    max_txt_v = -math.inf
    max_txt_a = -math.inf
    for txt in txts:
        txt_a = get_arousal_score(txt.split(" "))[0]
        txt_v = get_valence_score(txt.split(" "))[0]
        max_txt_v = max(txt_a, max_txt_v)
        max_txt_a = max(txt_v, max_txt_a)
    txt_v = get_valence_score(txt.split(" "))[0]
    txt_a = get_valence_score(txt.split(" "))[0]
    # pt = positiveStrategy.estimate(0, [crt_lenght, v, a])
    # print(pt)
    arm = np.argmax([positiveStrategy.estimate(0,[crt_lenght,txt_v,txt_a]),positiveStrategy.estimate(1,[crt_lenght,txt_v,txt_a]),\
                     positiveStrategy.estimate(2,[crt_lenght,txt_v,txt_a])])
    print("choice of arm is: ",arm )
    crt_lenght += 1
    tmps,_ = topk_surprise_sampling_one(model,tokenizer,txts,b_factors[arm])
    txts = tmps

    print(len(txts))
    if crt_lenght >8:
        break
rankee = []
already = []
for txt in txts:
    # v = get_valence_score(txt.split(" "))[0]

    words = txt.split(" ")
    a = get_valence_score(words)[0]/len(words)

    if a not in already:
        rankee.append((txt, a))
        already.append(a)
rankee.sort(key=lambda tup: tup[1], reverse=True)
print(rankee[:5])
# not 0 is correct