import pickle
import random
from IPython import embed
from utils import getStories, get_transformer
from config import Config
from valence_measure import get_valence_score, get_arousal_score
results = pickle.load(open("evaluation.p","rb"))
modes = ["small","medium","large","random","planned"]
model, tokenizer = get_transformer(Config.GPT2_finetuned_ROC)
win = {}

embed()
contexts, references = getStories(Config.story_path, story_num=50000, surprise_position=1)
modes = ["normal-small","normal-medium","normal-large","normal-random"]
import numpy as np
from tqdm import tqdm
all = []
def normal_gen(contexts, mode):
    if mode == "normal-small":
        beam = 10
    elif mode == "normal-medium":
        beam = 30
    elif mode == "normal-large":
        beam = 60
    elif mode == "normal-random":
        beam = random.choice([10,30,60])

    for context in tqdm(contexts[10000:10100]):
        context = context.strip(" ")
        input_ids = tokenizer(context, return_tensors="pt").input_ids

        sample_outputs = model.generate(input_ids=input_ids, max_length=8, typical_p=0.2, do_sample=True,
                                        num_return_sequences=beam)
        group = []
        for i, sample_output in enumerate(sample_outputs):
            text = tokenizer.decode(sample_output, skip_special_tokens=True)
            # embed()
            words = text.split(" ")
            # words = context.split(" ") + word
            a = get_valence_score(words)[0]
        group.append((mode,text,a))
        group.sort(key=lambda tup: tup[2], reverse=True)
        all.append(group[0])
    return all


for mode in modes:

    rt = normal_gen(contexts,mode)
    for cnt in range(100):
        results[cnt].append(rt[cnt])


for result in results:
    for num, tup in enumerate(result):
        if tup[0] not in win:
            win[tup[0]] = num
        else:
            win[tup[0]] += num
print(win)

arousal = {}
for result in results:
    for num, tup in enumerate(result):
        if tup[0] not in arousal:
            arousal[tup[0]] = tup[2]
        else:
            arousal[tup[0]] += tup[2]

print(arousal)