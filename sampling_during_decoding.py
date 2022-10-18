import os, sys

from utils import top_k_top_p_filtering, get_transformer
import pickle
from config import Config
import torch
from torch.nn import functional as F
import string
from nltk import tokenize
# model, tokenizer = get_transformer(Config.GPT2_finetuned_ROC)
from IPython import embed
punctuations = list(string.punctuation) + ['0','1','2','3','4','5','6','7','8','9'] + ['B','C',]
def wrapup_input(input,mode):
    if mode == "original":
        print('aba')
    elif mode == "prompt":
        prompt = '[title]' + '[story] '+ input
    return prompt

def count_punct(st):
    cnt = 0
    for i in st:
        if i in punctuations:
            cnt +=1
    return cnt

def get_storylines(prompt, crt_buget, model, tokenizer, branch_factor,device):
    '''
        score'(y_t, W_t | y<t) = score(y_t|y<t) + lambda * max(0, max cos_sim(y_t,W))
    '''


    wrappedup_model_input = wrapup_input(prompt,'prompt')
    max_len = 50
    input_ids = tokenizer.encode(wrappedup_model_input, return_tensors='pt')

    # model.cuda()
    # model.eval()

    input_ids = input_ids.to(device)
    cur_len = len(input_ids)
    input_tokens = [input_ids]
    valid = True
    if count_punct(prompt)>1:
        return [prompt]
    if crt_buget<0:
        beam_output = model.generate(input_ids=input_ids, max_length=50, do_sample=True,
                                        num_beams=5, pad_token_id=tokenizer.eos_token_id)
        txts = tokenizer.decode(beam_output [0], skip_special_tokens=True)

    else:
        decoded_tokens_list = expand_storylines(model, input_tokens, branch_factor)
        txts = []
        for i in range(len(decoded_tokens_list)):
            aba = tokenizer.decode(decoded_tokens_list[i][0], skip_special_tokens=True)

            txts.append(aba)

    return txts



def expand_storylines(model, input_tokens_list, topk):
    new_token_list = []
    check_list = []
    for input_tokens in input_tokens_list:
        input_ids = input_tokens
        logits = model(input_ids).logits[:, -1, :]
        ## Sample tokens
        logits = top_k_top_p_filtering(logits.squeeze(), top_k=100, top_p=0.9)  ###
        logits = F.softmax(logits, dim=-1)
        NUM = 1
        next_tokens_ids = torch.topk(logits, topk).indices
        # embed()
        tmp_token_list = []
        for next_token_id in next_tokens_ids:
            next_token_id = torch.unsqueeze(next_token_id,0)
            next_token_id = torch.unsqueeze(next_token_id, 0)
            tmp_token_list.append(torch.cat([input_ids, next_token_id], dim=-1))
        new_token_list += tmp_token_list
    return new_token_list

# model, tokenizer = get_transformer(Config.GPT2_finetuned_ROC)
# model.cuda()
# prompt = "Tom was eating hotdog on a train."
# ps=18
# txts = get_storylines(prompt, 10000, model, tokenizer,ps, device = torch.device("cuda"))
# print(txts)
# with open('storySpace_18^4.pickle', 'wb') as handle:
#     pickle.dump(txts, handle)
#
# ps=[10,10,100,10]
# txts = get_storylines(prompt, model, tokenizer,ps)
# with open('storySpace_100_2.pickle', 'wb') as handle:
#     pickle.dump(txts, handle)
#
# ps=[10,10,10,100]
# txts = get_storylines(prompt, model, tokenizer,ps)
# with open('storySpace_100_3.pickle', 'wb') as handle:
#     pickle.dump(txts, handle)

# while cur_len < max_len:
#     logits = model(input_ids).logits[:, -1, :]
#     ## Sample tokens
#     logits = top_k_top_p_filtering(logits.squeeze(), top_k=100, top_p=0.9)  ###
#     logits = F.softmax(logits, dim=-1)
#     NUM = 1
#     embed()
#     # next_tokens = torch.topk(logits, 100).indices
#     # tokens = tokenizer.decode(next_tokens, skip_special_tokens=True).split(" ")
#     #
#     next_token = torch.unsqueeze(torch.multinomial(logits, NUM), 0)
#     input_ids = torch.cat([input_ids, next_token], dim=-1)
#     cur_len += 1