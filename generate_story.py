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
from utils import wrapup_input
from proof_of_concept_experiment import topk_surprise_sampling_one
from sampling_during_decoding import count_punct

contexts, references = getStories(Config.story_path, story_num=50000, surprise_position=1)

model, tokenizer = get_transformer(Config.GPT2_finetuned_ROC)
model.cuda()
prompt = "Josh had a parrot that talked."
stories = finish_story(prompt, model, tokenizer, 5)

for sts in stories:
    story = " ".join(sts)
    print(story)