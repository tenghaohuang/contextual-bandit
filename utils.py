from random import random

from nltk.corpus import stopwords
from nltk import tokenize
from nltk.tokenize import word_tokenize
from scipy import spatial
import numpy as np
from transformers import GPT2LMHeadModel, GPT2Tokenizer, AutoModelForCausalLM, AutoTokenizer
import torch
from torch.nn import functional as F
import jsonlines
import re
import json
import pandas as pd
import os
import pickle
import string
from datetime import datetime
from sklearn.metrics.pairwise import cosine_similarity
from IPython import embed
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from config import Config
SPECIAL_TOKENS = {"bos_token": "<|BOS|>",
                  "eos_token": "<|EOS|>",

                  "pad_token": "<|PAD|>",
                  "sep_token": "<|SEP|>"}

stop_words = set(stopwords.words('english'))

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def get_GPT2_sampling_mode(input_path):
    if "_inferences" in input_path:
        input_path = "/".join(input_path.split("/")[:-1])
    infocard_path = input_path+"/infocard.txt"
    attrs = read_jsonl_file(infocard_path)
    dict = {}
    for d in attrs:
        dict[list(d.keys())[0]] = d[list(d.keys())[0]]
    return dict["GPT2_sampling_method"]

def getStories(story_path,story_num=10,surprise_position=2):
    columns = ["storyid", "storytitle", "sentence1", "sentence2", "sentence3", "sentence4", "sentence5"]
    with open(story_path, 'rb') as handle:
        df = pickle.load(handle)
    # df = pd.read_csv(story_path, encoding="ISO-8859-1", names=columns, engine='c')
    df = df[1:]
    setting = []
    reference = []
    story_num = min(95000,story_num)-1

    setting_cols = columns[2:2+surprise_position]
    reference_cols = columns[2+surprise_position:]
    assert(2+surprise_position<len(columns))

    for i in range(2,3+story_num):
      i = i+1
      pt = ""
      for j in setting_cols:
        if j != setting_cols[-1]:
            pt = pt + df.loc[i][j] + " "
        else:
            pt = pt + df.loc[i][j] + " \n"
      ref = ""
      for j in reference_cols:
        if j != reference_cols[-1]:
            ref = ref + df.loc[i][j] + " "
        else:
            ref = ref + df.loc[i][j] + " \n"
      setting.append(pt.strip("\n"))
      reference.append(ref)
    return setting,reference

def getDeberta():

    tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-large-mnli")

    model = AutoModelForSequenceClassification.from_pretrained(
        "/playpen-ssd/tenghao/creative_writing/Finetuned_GPT2/antiComet_roberta/Tuned_Model/")
    return model, tokenizer

def getRoberta():
    # roberta = SentenceTransformer('sentence-transformers/nli-roberta-large')
    roberta = torch.hub.load('pytorch/fairseq', 'roberta.large.mnli', force_reload=True)
    roberta.cuda()
    roberta.eval()
    return roberta

def getContradictionScores(roberta, setting, continuation):

    tokens = roberta.encode(setting,continuation)
    value = roberta.predict('mnli', tokens).cpu().detach().numpy()
    value = [round(i,3) for i in value[0].tolist()]
    return value, continuation.capitalize()

def getRoberta():
    roberta = torch.hub.load('pytorch/fairseq', 'roberta.large.mnli')
    roberta.cuda()
    roberta.eval()
    return roberta


def get_xWants(first_stage, model, tokenizer, surprise_position):
    prompts = []
    xWants = []
    for setting_gene in first_stage:
        xWant = setting_gene[surprise_position - 1]['<|xWant|>']
        xWants.append(xWant)
        prompt = setting_gene[surprise_position - 1]['story']
        prompts.append(prompt)
        embed()

    return 0

def get_embedding_dict(path):
    emmbed_dict = {}
    with open(path, 'r') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], 'float32')
            emmbed_dict[word] = vector
    return emmbed_dict


def get_transformer(load_path=None):
    # SPECIAL_TOKENS = {"bos_token": "<|BOS|>",
    #                   "eos_token": "<|EOS|>",
    #                   "sep_token": "<|SEP|>"}
    SPECIAL_TOKENS = {"bos_token": "[title]",
                      "eos_token": "[end]",
                      "sep_token": "[story]"}
    tokenizer = AutoTokenizer.from_pretrained('gpt2-large')
    model = GPT2LMHeadModel.from_pretrained('gpt2-large')
    if load_path is not None:
        print(len(tokenizer))
        tokenizer.add_special_tokens(SPECIAL_TOKENS)
        model.resize_token_embeddings(len(tokenizer))
        print(len(tokenizer))
        model.load_state_dict(torch.load(load_path))
    return model, tokenizer


def wrapup_input(input,mode):
    if mode == "original":
        prompt = SPECIAL_TOKENS['bos_token'] + input + \
                 SPECIAL_TOKENS['sep_token']
    elif mode == "prompt":
        # prompt = '[title] ' +input+ ' [story] '
        prompt = '[title] [story] '+input
    return prompt


def non_typical_sampling(p, p_prime, alpha):
    zeros = torch.zeros(1, p.size(1) - p_prime.size(1))
    p_prime = torch.cat([p_prime, zeros], dim=-1)
    # top_p = torch.topk(p, 100).values
    # print(torch.mean(top_p))

    # top_p_prime = torch.topk(p_prime, 100).values
    # print(torch.mean(top_p_prime))
    # assert(False)
    # mean of the top 100 values is around 0.009
    new_p = p - alpha * p_prime
    new_p[new_p < 0] = 0
    return new_p


# def anti_paracomet_sample(p, sim, alpha):
#     new_p = p - alpha * sim
#     new_p[new_p < 0] = 0
#     return new_p
#
# def anti_paracomet_sample_logits(logits,sim, alpha):
#     new_l = logits - alpha * sim
#     return new_l

def remove_stop_words(text):
    word_tokens = word_tokenize(text)
    filtered_text = [w for w in word_tokens if not w.lower() in stop_words]
    filtered_text = [i for i in filtered_text if i != "like"]
    return filtered_text


def get_cos_similarity(model, tokenizer, src_word_list, paracomet_inf, embed_dict):
    paracomet_inf = [remove_stop_words(i) for i in paracomet_inf]
    word_cloud = []
    for l in paracomet_inf:
        word_cloud += l
    # print(word_cloud)
    word_cloud = list(set(word_cloud))
    cloud_embeds = []
    for word in word_cloud:

        # encoded = tokenizer.encode(word,return_tensors='pt')
        # print(word)
        # print(model.transformer.wte.weight[encoded,:].size())
        if word in embed_dict:
            cloud_embeds.append(embed_dict[word])
    cloud_embeds = [i for i in cloud_embeds if i != []]
    rt = []
    for src_word_index in src_word_list[0]:
        # encoded_src_word = src_word_index.repeat(1,len(word_cloud))
        # src_word_embed = model.transformer.wte.weight[encoded_src_word,:]
        decoded = tokenizer.decode(src_word_index)
        if decoded.lower() in embed_dict:
            decoded_embed = embed_dict[decoded.lower()]
        else:
            rt.append(0)
            continue
        # cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
        cloud_cos_sim = []
        for tgt_embed in cloud_embeds:
            cloud_cos_sim.append(1 - spatial.distance.cosine(decoded_embed, tgt_embed))
        max_v = max(cloud_cos_sim)
        # print(max_v)
        rt.append(max_v)
    return rt




def create_unique_folder(outputpath):
    dt_string = get_date_sting()
    outputpath = os.path.join(outputpath,dt_string)
    os.mkdir(outputpath)
    return outputpath

def get_date_sting():
    now = datetime.now()
    dt_string = str(now.strftime("%d-%m-%Y-%H:%M:%S"))
    return dt_string

def write_jsonl_file(items, path,mode="w"):
    if type(items[0]) == list:
        tmp = []
        for item in items:
            tmp+=item
            tmp.append("\n")
        items = tmp
    with jsonlines.open(path, mode) as writer:
        writer.write_all(items)

def read_jsonl_file(filename):
    with open(filename, 'r') as json_file:
        json_list = list(json_file)
    anti_paracomet = []
    for json_str in json_list:
        anti_paracomet.append(json.loads(json_str))
    return anti_paracomet


def get_top_choices(cur_len,pre,post):
    pre_top = torch.topk(pre, 100).indices.tolist()
    post_top = torch.topk(post, 100).indices.tolist()
    rt = {"cur_len":cur_len,"original":pre_top,"processed":post_top}
    return rt

def split_sentences(corpus):
    return tokenize.sent_tokenize(corpus)




def finish_story(prompt, model, tokenizer, beam_size,num_story_return, surprise_position = -1):

    prompt = prompt.strip("\n")
    prompt = prompt.strip(" ")
    if prompt[-1] not in string.punctuation:
        prompt = prompt +". "
    warpped_prompt = wrapup_input(prompt,"prompt")
    input_ids = tokenizer(warpped_prompt, return_tensors="pt").input_ids.cuda()
    sample_outputs = model.generate(input_ids=input_ids, max_length=100, do_sample=True, beam_size=beam_size,num_return_sequences=num_story_return)
    conts = []
    texts = []
    for i, sample_output in enumerate(sample_outputs):
        text = tokenizer.decode(sample_output, skip_special_tokens=True)
        sts = tokenize.sent_tokenize(text)
        # found an exception: " I was talking to my crush today. \
        # She continued to complain about guys flirting with her.", "
        # ^ the continuation is invalid
        texts.append(sts)
        # if surprise_position+1>len(sts):
        #     continuation = "None."
        #     sts.append(continuation)
        # else:
        #     continuation = sts[surprise_position]
        #
        # if surprise_position>0:
        #     conts.append(continuation)
        #     texts.append(sts[:surprise_position+1])
        # else:
        #     conts.append(sts[surprise_position:])
        #     texts.append(" ".join(sts))
    return  texts

def finish_stories(prompts, model, tokenizer, num_story_return, num_st_return=-1 ):
    rt = []
    for num, prompt in enumerate(prompts):
        prompt = prompt.strip("\n")
        if prompt[-1] not in string.punctuation:
            prompt = prompt +". "
        warpped_prompt = wrapup_input(prompt,"")
        input_ids = tokenizer(warpped_prompt, return_tensors="pt").input_ids
        sample_outputs = model.generate(input_ids=input_ids, max_length=100, do_sample=True, num_return_sequences=num_story_return)
        texts = []
        for i, sample_output in enumerate(sample_outputs):
            text = tokenizer.decode(sample_output, skip_special_tokens=True)
            text = split_sentences(text)
            if num_st_return != -1:
                assert(num_st_return+1>len(text))
                text = text[:num_st_return+1]
            rt.append(text)
    return rt

def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (vocabulary size)
            top_k >0: keep only top k tokens with highest probability (top-k filtering).
            top_p >0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
    """

    assert logits.dim() == 1  # batch size 1 for now - could be updated for more but the code would be less clear
    top_k = min(top_k, logits.size(-1))  # Safety check
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # indices_to_remove = torch.zeros_like(logits, dtype=torch.uint8).scatter_(dim=-1, index=sorted_indices, src=sorted_indices_to_remove )

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value

    return logits

# def model_generate(args, result_path,trained_model, trained_tokenizer, model, tokenizer, \
#                    model_input, setting_id, story_id, s3=None, paracomet_inf=None, sampling_mode="non-typical", max_len=50):
#     embed_dict = get_embedding_dict(args.glove_path)
#     wrappedup_model_input = wrapup_input(model_input,'original')
#
#     input_ids = trained_tokenizer.encode(wrappedup_model_input, return_tensors='pt')
#     cur_len = 0
#     # Get logits from last layer
#     winners = []
#
#     while cur_len < max_len:
#         inspect_dumpee = []
#         last_layer_logits = trained_model(input_ids).logits[:, -1, :]
#
#         if sampling_mode == "normal":
#             top_logits = top_k_top_p_filtering(last_layer_logits, top_k=100, top_p=1.0)
#             probabilities = F.softmax(top_logits, dim=-1)
#         elif sampling_mode == "non-typical":
#             p_prime = next_token_prob(model, tokenizer, s3)
#             alpha = 0.8
#             p = F.softmax(last_layer_logits, dim=-1)
#             probabilities = non_typical_sampling(p, p_prime, alpha)
#
#         elif sampling_mode == "anti_paracomet_logits":
#             alpha = 30
#             top_logits = top_k_top_p_filtering(last_layer_logits, top_k=100, top_p=1.0)
#             # the average velaue of top 100 tokens is 13
#             constructed = torch.zeros(top_logits.size(1))
#             src_word_indices = torch.topk(top_logits, 100).indices
#             cos_sims = gecos_similarity(trained_model, trained_tokenizer, src_word_indices, paracomet_inf, embed_dict)
#             for num, index in enumerate(src_word_indices[0].tolist()):
#                 constructed[index] = cos_sims[num]
#             processed_logits = anti_paracomet_sample_logits(top_logits, constructed, alpha)
#
#             # try to inpect how top choices change after anti_sample
#             logits_tup = get_top_choices(cur_len,top_logits,processed_logits)
#             inspect_dumpee.append(logits_tup)
#
#             processed_logits = top_k_top_p_filtering(processed_logits, top_k=100, top_p=1.0)
#             probabilities = F.softmax(processed_logits, dim=-1)
#
#         elif sampling_mode == "anti_paracomet":
#             alpha = 0.8
#             top_logits = top_k_top_p_filtering(last_layer_logits, top_k=100, top_p=1.0)
#             pre_probabilities = F.softmax(top_logits, dim=-1)
#             src_word_indices = torch.topk(pre_probabilities, 100).indices
#             cos_sims = get_cos_similarity(trained_model, trained_tokenizer, src_word_indices, paracomet_inf, embed_dict)
#
#             # try to inpect how top choices change after anti_sample
#
#
#             constructed = torch.zeros(pre_probabilities.size(1))
#             for num, index in enumerate(src_word_indices[0].tolist()):
#                 constructed[index] = cos_sims[num]
#             probabilities = anti_paracomet_sample(pre_probabilities, constructed, alpha)
#
#             logits_tup = get_top_choices(cur_len,pre_probabilities,probabilities)
#             inspect_dumpee.append(logits_tup)
#         else:
#             assert (False)
#         #write_inspection_file
#         inspect_folder = result_path+"/setting"+str(setting_id)+"/story"+str(story_id)
#         isExist = os.path.exists(inspect_folder)
#         if not isExist:
#             os.makedirs(inspect_folder)
#         # print(inspect_dumpee)
#         write_jsonl_file(inspect_dumpee,inspect_folder+\
#                          "/token"+str(cur_len)+".jsonl",mode="a")
#
#         NUM = 1
#         next_token = torch.multinomial(probabilities, NUM)
#         input_ids = torch.cat([input_ids, next_token], dim=-1)
#         cur_len += 1
#
#     return input_ids


def next_token_prob(gpt2, tokenizer, s3):
    input_ids = tokenizer.encode(s3, return_tensors='pt')
    logits = gpt2(input_ids).logits[:, -1, :]
    p_prime = F.softmax(logits, dim=-1)
    return p_prime
