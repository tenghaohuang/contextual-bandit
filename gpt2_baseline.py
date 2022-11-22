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
prompt = "I was talking to my crush today."
rt = finish_story(prompt, model, tokenizer,5)
print(rt)
