import pickle
import random
from IPython import embed
from utils import getStories, get_transformer
from config import Config
from valence_measure import get_valence_score, get_arousal_score

modes = ["small","medium","large","random","planned"]



# model, tokenizer = get_transformer(Config.GPT2_finetuned_ROC)
# contexts, references = getStories(Config.story_path, story_num=50000, surprise_position=1)
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


# for mode in modes:
#
#     rt = normal_gen(contexts,mode)
#     for cnt in range(100):
#         results[cnt].append(rt[cnt])
#
def get_ascore(txt):
    words = txt.split(" ")
    a = get_arousal_score(words)[0]
    return a


def get_result(file):
    win = {}
    print(file)
    results = pickle.load(open(file, "rb"))
    for result in results:
        new_result = [(tup[0],get_ascore(tup[1])) for tup in result]
        # print(new_result)
        # assert(False)
        new_result.sort(key=lambda tup: tup[1], reverse=True)
        for num, tup in enumerate(new_result):

            if tup[0] not in win:
                win[tup[0]] = 1/(num+1)
            else:
                win[tup[0]] += 1/(num+1)

    print(win)

    arousal = {}
    for result in results:
        for num, tup in enumerate(result):
            if tup[0] not in arousal:
                arousal[tup[0]] = tup[2]
            else:
                arousal[tup[0]] += tup[2]

    print(arousal)
file = "evaluation_4.p"
get_result(file)
file = "evaluation_6.p"
get_result(file)
file = "evaluation_8.p"
get_result(file)