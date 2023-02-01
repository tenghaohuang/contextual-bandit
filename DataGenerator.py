import math
import random

import numpy as np

from sampling_during_decoding import get_storylines
from utils import get_transformer
from config import Config
from valence_measure import get_valence_score, get_arousal_score
from nltk.tokenize import word_tokenize
from ppl_sentence_level import get_story_ppl, process_story
import string
from GLTR import LM
import torch
from scipy.special import softmax
from IPython import embed
class DataGenerator():
    """
    Generate badit data.

    Defaults:
    K=2 arms
    D=2 features/arm
    """
    def __init__(self,K=2,D=2,feature_type='binary',reward_type='binary'):
        
        self.D = D # dimension of the feature vector
        self.K = K # number of bandits
        self.reward_type = reward_type
        self.feature_type = feature_type
        self.means = np.random.normal(size=self.K)
        self.stds = 1 + 2*np.random.rand(self.K)
        self.device =  torch.device("cuda")
        # generate the weight vectors.  initialize estimate of feature
        # importance for each arm's d features
        self.generate_weight_vectors()
        self.model, self.tokenizer = get_transformer(Config.GPT2_finetuned_ROC)
        self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        self.model.resize_token_embeddings(len(self.tokenizer))
        self.model.to(self.device)
        self.model.eval()
        # self.LM = LM(self.model, self.tokenizer)
        self.LM = ""
        self.topk = 10
        self.branch_factors = [10, 30, 60]
        self.evaluating = False

    def generate_weight_vectors(self,loc=0.0,scale=1.0):
        self.W = np.random.normal(loc=loc,scale=scale,size=(self.K,self.D))
        #self.W = np.ones((self.K,self.D))

    def group_bucket(self, scores, anchors, b):
        l = []
        start = 0
        for num in range(len(anchors)):
            aba = scores[start:start + min(b, anchors[num])]
            if type(aba) is not list:
                aba = [aba]
            l += aba
            start += anchors[num]
        return l

    def get_topk_reward_each_bucket(self, buckets, budget, crt_length, previous_pivot):
        sample_rewards = []
        best_per_bucket = []
        # branch_facts = [True if budget > b else False for b in self.branch_factors]

        # print(tup)
        # embed()#look at perplex
        for num in range(len(buckets)):
            b = self.branch_factors[num]
            buckets[num].sort(key=lambda tup: tup[1], reverse=True)
            reward = np.exp(buckets[num][0][1] - 0.0001 * b - previous_pivot)
            sample_rewards.append(reward)
        # sample_rewards = [5 * i / sum(sample_rewards) for i in sample_rewards]
        sample_rewards = np.e ** (5*softmax(sample_rewards)) - np.e

        for num in range(len(buckets)):
            best_per_bucket.append((buckets[num][0][0], sample_rewards[num], \
                                    buckets[num][0][-1]))

        return sample_rewards, best_per_bucket
    # def get_topk_reward_each_bucket(self, buckets, budget, crt_length):
    #     sample_rewards = []
    #     best_per_bucket = []
    #     # branch_facts = [True if budget > b else False for b in self.branch_factors]
    #
    #     # print(tup)
    #     # embed()#look at perplex
    #     for num in range(len(buckets)):
    #
    #
    #         b = self.branch_factors[num]
    #
    #
    #         buckets[num].sort(key=lambda tup: tup[1], reverse=True)
    #         sample_rewards.append(buckets[num][0][1]-0.0010*b)
    #         best_per_bucket.append((buckets[num][0][0],buckets[num][0][1]-0.0010*b,\
    #                                 buckets[num][0][-1]))
    #
    #     return sample_rewards, best_per_bucket

    def put_buckets(self, buckets, txts_at_timestep):
        for num, b in enumerate(self.branch_factors):
            buckets[num]+=txts_at_timestep[:b]
        return buckets

    def generate_gpt_topk_example(self,text_prompts_ppls, crt_length, crt_budget):
        txts_at_timestep = []

        overall_features = []

        buckets = [[],[],[]]
        max_txt_v = -math.inf
        max_txt_a = -math.inf
        # print(text_prompts_ppls)
        # print(crt_length)
        for txt_prompt_ppl in text_prompts_ppls:

            txt = txt_prompt_ppl[0]

            txt_a = get_arousal_score(txt.split(" "))[0]
            txt_v = get_valence_score(txt.split(" "))[0]
            max_txt_v = max(txt_a,max_txt_v)
            max_txt_a = max(txt_v, max_txt_a)
            rt_txts_ppls = get_storylines(txt_prompt_ppl, crt_length, self.model, self.tokenizer, self.LM, self.branch_factors[-1], device=torch.device("cuda"))
            # embed()
            crt_appendee = []
            for tmp in rt_txts_ppls:
                appendee_txt = tmp[0]
                appendee_txt_ppl = tmp[1]
                # v = get_valence_score(tmp.split(" "))[0]
                a = get_arousal_score(appendee_txt.split(" "))[0]
                gem = (appendee_txt, a - 0.003*appendee_txt_ppl, txt_v, txt_a, appendee_txt_ppl)
                txts_at_timestep.append(gem)
                crt_appendee.append(gem)
            buckets = self.put_buckets(buckets, crt_appendee)
        overall_features.append((crt_length, max_txt_v, max_txt_a, txt_prompt_ppl[1]))
        prompt_sample_rewards, best_per_bucket = self.get_topk_reward_each_bucket(buckets, crt_budget, crt_length, previous_pivot=max_txt_a)

        return txts_at_timestep, overall_features, np.asarray(prompt_sample_rewards), best_per_bucket, buckets


    def filter_by_perplexity(self, candidate_list,crt_length):
        rt_l = []
        for story in candidate_list:
            context_sentences = process_story(story)
            ppl_scores = get_story_ppl(context_sentences, self.tokenizer, self.model, self.device)
            if ppl_scores[-1]<(crt_length)*1000:
                rt_l.append(story)
        if len(rt_l)<int(len(candidate_list)/3):
            rt_l = []
        return rt_l



    # the bandit algorithm
    def run_bandit_alg(self,true_rewards,CTRs_that_generated_data,choice_func):
        num_samples,K = true_rewards.shape
        observed_data = np.zeros((K,2))
        # seed the estimated params
        prior_a = 1. # aka successes
        prior_b = 1. # aka failures
        observed_data[:,0] += prior_a # allocating the initial conditions
        observed_data[:,1] += prior_b
        regret = np.zeros(num_samples)

        for i in range(0,num_samples):
            # pulling a lever & updating observed_data
            this_choice = choice_func(observed_data)

            # update parameters
            if true_rewards[i,this_choice] == 1:
                update_ind = 0
            else:
                update_ind = 1

            observed_data[this_choice,update_ind] += 1

            # updated expected regret
            regret[i] = np.max(CTRs_that_generated_data[i,:]) - CTRs_that_generated_data[i,this_choice]

        cum_regret = np.cumsum(regret)

        return cum_regret
