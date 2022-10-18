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
import torch
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


    def get_reward_each_bucket(self, tup, budget, anchors, crt_length):
        best_per_bucket = []
        val_scores, arousal_scores = tup
        sample_rewards = []
        branch_facts = [True if budget > b else False for b in self.branch_factors]


        for num, b in enumerate(self.branch_factors):

            # embed()
            if branch_facts[num] is False:
                arousal_scores_tmp = self.group_bucket(arousal_scores, anchors, budget)
                arousal_scores_tmp.sort(key=lambda tup: tup[0], reverse=True)
                sample_rewards.append(arousal_scores_tmp[0][0])
                best_per_bucket.append(arousal_scores_tmp[:2])
            else:
                arousal_scores_tmp = self.group_bucket(arousal_scores, anchors, b)
                arousal_scores_tmp.sort(key=lambda tup: tup[0], reverse=True)
                sample_rewards.append(arousal_scores_tmp[0][0])
                best_per_bucket.append(arousal_scores_tmp[:2])
        # if crt_length > 3:
        #
        return sample_rewards, best_per_bucket

    def get_topk_reward_each_bucket(self, buckets, budget, crt_length):
        sample_rewards = []
        best_per_bucket = []
        # branch_facts = [True if budget > b else False for b in self.branch_factors]

        # print(tup)
        for num in range(len(buckets)):

            # embed()
            b = self.branch_factors[num]

            buckets[num].sort(key=lambda tup: tup[1], reverse=True)
            sample_rewards.append(buckets[num][0][1]-0.0025*b)
            best_per_bucket.append((buckets[num][0][0],buckets[num][0][1]-0.0025*b))

        return sample_rewards, best_per_bucket

    def put_buckets(self, buckets, txts_at_timestep):
        for num, b in enumerate(self.branch_factors):
            buckets[num]+=txts_at_timestep[:b]
        return buckets

    def generate_gpt_topk_example(self,text_prompts, crt_length, crt_budget):
        txts_at_timestep = []

        overall_features = []

        buckets = [[]]*len(self.branch_factors)
        max_txt_v = -math.inf
        max_txt_a = -math.inf
        for txt in text_prompts:
            txt_a = get_arousal_score(txt.split(" "))[0]
            txt_v = get_valence_score(txt.split(" "))[0]
            max_txt_v = max(txt_a,max_txt_v)
            max_txt_a = max(txt_v, max_txt_a)
            tmps = get_storylines(txt, crt_budget, self.model, self.tokenizer, self.branch_factors[-1], device=torch.device("cuda"))
            # embed()
            for tmp in tmps:
                # v = get_valence_score(tmp.split(" "))[0]
                a = get_arousal_score(tmp.split(" "))[0]
                txts_at_timestep.append((tmp, a, txt_v, txt_a))

            buckets = self.put_buckets(buckets, txts_at_timestep)
        overall_features.append((crt_length, max_txt_v, max_txt_a))
        prompt_sample_rewards, best_per_bucket = self.get_topk_reward_each_bucket(buckets, crt_budget, crt_length)

        return txts_at_timestep, overall_features, np.asarray(prompt_sample_rewards), best_per_bucket

    def generate_gpt_examples(self, text_prompts, crt_length, crt_budget):
        """
        :param n:
        :return:
            sample_features: a list of (crt_budget, crt_valence, crt_arousal, ctr_perplexity)

            sample_rewards:  a list of (arousal) tuples
        """

        overall_lists = []
        overall_rewards = []
        overall_sample_features = []

        anchors = []
        candidate_list = []
        prompt_val_scores = []
        prompt_arousal_scores = []
        prompts_lens = []
        for id, prompt in enumerate(text_prompts):

            prompt_tokens = word_tokenize(prompt)
            # overall_sample_features.append((crt_length, crt_budget, prompt_val_score, prompt_arousal_score, ppls[id].cpu().detach().numpy()))

            if not self.evaluating:
                candidate_list_, local_validates_infos = get_storylines(prompt, crt_budget, self.model, self.tokenizer, self.branch_factors[-1],self.device)
            #filter out random shit

            # if crt_length == 3:
            #     candidate_list_ = self.filter_by_perplexity(candidate_list_, crt_length)
            if candidate_list_ == []:
                continue
            candidate_list += candidate_list_
            prompts_lens += [len(prompt)]*len(candidate_list_)
            prompt_val_scores += [get_valence_score(prompt_tokens)[0]]*len(candidate_list_)
            prompt_arousal_scores += [get_valence_score(prompt_tokens)[0]]*len(candidate_list_)
            anchors.append(len(candidate_list_))
        overall_sample_features.append(
            (crt_length, crt_budget, max(prompt_val_scores), max(prompt_arousal_scores)))


        # cut the original length

        curated_candidate_list = [i[prompts_lens[num]:] for num,i in enumerate(candidate_list) ]
        candidate_list_tokens = [word_tokenize(candidate) for candidate in curated_candidate_list]
        val_scores = [(prompt_val_scores[num] + get_valence_score(i)[0],num) if get_valence_score(i)[0] is not None else (prompt_val_scores,num) \
                          for num, i in enumerate(candidate_list_tokens)]
        arousal_scores = [(prompt_arousal_scores[num] + get_arousal_score(i)[0], num) if get_arousal_score(i)[0] is not None else (prompt_arousal_scores, num) \
                              for num, i in enumerate(candidate_list_tokens)]

        prompt_sample_rewards, best_per_bucket = self.get_reward_each_bucket((val_scores,arousal_scores),crt_budget, anchors, crt_length)

        best_gen_per_bucket = []
        for buckets in best_per_bucket:
            tmp = []
            for score, rank in buckets:
                tmp.append(candidate_list[int(rank)])
            best_gen_per_bucket.append((score,tmp))

        overall_rewards.append(prompt_sample_rewards)
        overall_lists += candidate_list


        # overall_best_per_bucket.sort(key=lambda tup: tup[0], reverse=True)
        return overall_lists, overall_sample_features, np.asarray(overall_rewards),best_gen_per_bucket,anchors

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

    def generate_samples(self,n=1000):
        # the sample feature vectors X are only binary

        if self.feature_type == 'binary':
            X = np.random.randint(0,2,size=(n,self.D))
        elif self.feature_type == 'integer':
            X = np.random.randint(0,5,size=(n,self.D))
        
        # the rewards are functions of the inner products of the
        # feature vectors with (current) weight estimates       
        IP = np.dot(X,self.W.T)

        # now get the rewards
        if self.reward_type == 'binary':
            R = ((np.sign(np.random.normal(self.means + IP,self.stds)) + 1) / 2).astype(int)
        elif self.reward_type == 'positive':
            #R = np.random.lognormal(self.means + IP,self.stds)
            R = np.abs(np.random.normal(self.means + IP,self.stds))
        elif self.reward_type == 'mixed':
            R = (np.sign(np.random.normal(self.means + IP,self.stds)) + 1) / 2
            R *= np.random.lognormal(self.means + IP,self.stds)

        return X,R

    # generate all bernoulli rewards ahead of time
    def generate_bernoulli_bandit_data(self,num_samples):
        #initialize parameter estimates
        CTRs_that_generated_data = np.tile(np.random.rand(self.K),(num_samples,1))
        #did the trial succeed?
        true_rewards = np.random.rand(num_samples,self.K) < CTRs_that_generated_data
        return true_rewards,CTRs_that_generated_data

    # Thompson Sampling
    # basic idea: samples from distribution and compares those values for the arms instead
    # http://www.economics.uci.edu/~ivan/asmb.874.pdf
    # http://camdp.com/blogs/multi-armed-bandits
    def thompson_sampling(self,observed_data):
        return np.argmax( np.random.beta(observed_data[:,0], observed_data[:,1]) )


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
