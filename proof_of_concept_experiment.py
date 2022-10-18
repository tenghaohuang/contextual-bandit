from config import Config
from sampling_during_decoding import get_storylines
from utils import get_transformer
from valence_measure import get_valence_score, get_arousal_score
import torch
import pickle
# model, tokenizer = get_transformer(Config.GPT2_finetuned_ROC)
# model.cuda()
# model.eval()

def topk_surprise_sampling(model, tokenizer, prompt, branch_factors):
    txts = [prompt]

    for b_at_timestep in branch_factors:
        txts_at_timestep = []
        for txt in txts:
            tmps = get_storylines(txt, 10000, model, tokenizer, b_at_timestep, device=torch.device("cuda"))
            for tmp in tmps:
                v = get_valence_score(tmp.split(" "))[0]
                a = get_arousal_score(tmp.split(" "))[0]
                txts_at_timestep.append((tmp, a))
        txts_at_timestep.sort(key=lambda tup: tup[1], reverse=True)
        txts_at_timestep = txts_at_timestep[0]
        txts = [i[0] for i in txts_at_timestep]
    return txts, txts_at_timestep

def topk_surprise_sampling_one(model, tokenizer, txts, b_at_timestep):

    txts_at_timestep = []
    for txt in txts:
        tmps = get_storylines(txt, 10000, model, tokenizer, b_at_timestep, device=torch.device("cuda"))
        for tmp in tmps:
            v = get_valence_score(tmp.split(" "))[0]
            a = get_arousal_score(tmp.split(" "))[0]
            txts_at_timestep.append((tmp, a))
    txts_at_timestep.sort(key=lambda tup: tup[1], reverse=True)
    txts_at_timestep = txts_at_timestep[:b_at_timestep]
    txts = [i[0] for i in txts_at_timestep]
    return txts, txts_at_timestep

# prompt = "Tom is eating a hotdog on a train."
# # Experiment 1
branch_factors_1 = [20,20,20,20]
# 1220
# rt = topk_surprise_sampling(model,tokenizer,prompt,branch_factors_1)
# # with open('storySpace_20-20-20-20.pickle', 'wb') as handle:
# #     pickle.dump(rt, handle)
# print(rt[1][:3])
# print("guaguagua")
#
#
branch_factors_2 = [20,60, 7, 20]
# rt = topk_surprise_sampling(model,tokenizer,prompt,branch_factors_2)
# # with open('storySpace_20-60-7-2.pickle', 'wb') as handle:
# #     pickle.dump(rt, handle)
# print(rt[1][:3])
# print("guaguagua")
#
# branch_factors_3 = [20, 7, 60, 20]
# rt = topk_surprise_sampling(model,tokenizer,prompt,branch_factors_3)
# # with open('storySpace_2-7-60-20.pickle', 'wb') as handle:
# #     pickle.dump(rt, handle)
# print(rt[1][:3])
# print("guaguagua")
#
# branch_factors_4 = [7, 20, 60, 20]
# rt = topk_surprise_sampling(model,tokenizer,prompt,branch_factors_4)
# # with open('storySpace_7-20-60-2.pickle', 'wb') as handle:
# #     pickle.dump(rt, handle)
# print(rt[1][:3])
# print("guaguagua")
#
# branch_factors_5 = [7, 60, 20, 20]
# rt = topk_surprise_sampling(model,tokenizer,prompt,branch_factors_1)
# with open('storySpace_7-60-20-2.pickle', 'wb') as handle:
#     pickle.dump(rt, handle)
# print(rt[1][:3])
# print("guaguagua")
