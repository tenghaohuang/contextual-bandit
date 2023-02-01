import math
import random

import DataGenerator
from utils import getStories, get_transformer, finish_story
from config import Config
# import Simulator
from IPython import embed
from valence_measure import get_valence_score, get_arousal_score
import pickle
import numpy as np
import DataGenerator
from utils import get_transformer
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
model, tokenizer = get_transformer(Config.GPT2_finetuned_ROC)
model.cuda()
my_results = pickle.load(open("/nas/luka-group/tenghao/tenghaoh/creative_writing/surprise_bandit/evaluation_4_planned.p",'rb'))
# prompt = "I was talking to my crush today."

import tqdm
print(my_results)
def get_ascore(txt):
    words = txt.split(" ")
    a = get_arousal_score(words)[0]/len(words)
    return a

def get_txt_a_pair(type, txt):
    return (type, txt,get_ascore(txt))
all = []
for num, context in enumerate(contexts[10000:10020]):
    mine = my_results[num]
    base_10 = finish_story(context, model, tokenizer,beam_size=10,num_story_return=1)
    base_10 = " ".join(base_10[0])
    base_30 = finish_story(context, model, tokenizer,beam_size=30,num_story_return=1)
    base_30 = " ".join(base_30[0])
    base_60 = finish_story(context, model, tokenizer,beam_size=60,num_story_return=1)
    base_60 = " ".join(base_60[0])
    savee = [mine,get_txt_a_pair('base_10',base_10),get_txt_a_pair('base_30',base_30),get_txt_a_pair('base_60',base_60)]
    print(savee)
    savee.sort(key=lambda tup: tup[2], reverse=True)
    all.append(savee)
pickle.dump(all, open("everything_1_31.p","wb"))




win = {}

# print(planned_results)
total = []
for num, result in enumerate(all):

    # print(new_result)
    # assert(False)
    for num, tup in enumerate(result):

        if tup[0] not in win:
            win[tup[0]] = 1/(num+1)
        else:
            win[tup[0]] += 1/(num+1)
print(win)