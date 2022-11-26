#from mpltools import style # uncomment for prettier plots
#style.use(['ggplot'])

import DataGenerator
import PositiveStrategy
import Simulator
from utils import getStories, wrapup_input
from config import Config
from IPython import embed
import numpy as np
import pickle
from tqdm import tqdm
arms=3
features=4
rewardType='positive'
#rewardType='binary'
featureType='integer'
#featureType='binary'

# define number of samples and number of choices
contexts, references = getStories(Config.story_path, story_num=50000, surprise_position=1)
dg = DataGenerator.DataGenerator(arms, features, feature_type=featureType, reward_type=rewardType)
positiveStrategy = PositiveStrategy.PositiveStrategy(arms, features)
# print(positiveStrategy)
simulator = Simulator.Simulator(positiveStrategy)
print("surprise_bandit_epoch_aba1000")
for epoch in range(10):
    num_samples = 20001
    num_batches = 100
    num_experiments = 1

    total_regret = []
    total_rmse = []
    total_budget = 200000
    records = []
    for id in tqdm(range(0, 3000)):
        print("experiment: %d" % id)

        # appropriate model for reward_type='positive' or reward_type='mixed'



        # assert(False)
        previous_rmse = 0.
        #     if previous_rmse == 0:
        #         initial_rmse = rmse[0][-1]
        crt_budget = total_budget
        crt_length = 0
        print(contexts[id])
        if contexts[id].endswith(" "):
            prompts_ppls = [[wrapup_input(contexts[id][:-1],'prompt'),1]]
        else:
            prompts_ppls = [[wrapup_input(contexts[id],'prompt'),1]]
        training_examples = []
        while True:
            crt_length += 1
            # TODO: it is now 0 for batch punishment
            (txts_at_timestep, overall_sample_features, overall_rewards, best_per_bucket, buckets) = dg.generate_gpt_topk_example(prompts_ppls,crt_length, crt_budget)
            # if len(txts_at_timestep)==len(prompts_ppls):
            #     break
            if crt_length>10:
                break
            print("the length of generated prompts are", len(txts_at_timestep))
            regret, rmse, armChoice = simulator.simulate(overall_sample_features, overall_rewards, dg.W)
            gold_choice = np.argmax(overall_rewards)
            print("the choice is ", armChoice)
            print("the gold choice is ", gold_choice)
            # txts_at_timestep = buckets[armChoice]
            # txts_at_timestep.sort(key=lambda tup: tup[1], reverse=True)
            # txts_at_timestep = txts_at_timestep[:dg.branch_factors[armChoice]]
            # # embed()
            # print("the length of filtered prompts are", len(txts_at_timestep))
            # prompts_ppls = [(i[0],i[-1]) for i in txts_at_timestep]
            # crt_budget = int(crt_budget/dg.branch_factors[armChoice])
            txts_at_timestep = buckets[gold_choice]
            txts_at_timestep.sort(key=lambda tup: tup[1], reverse=True)
            txts_at_timestep = txts_at_timestep[:dg.branch_factors[gold_choice]]
            # embed()
            print("the length of filtered prompts are", len(txts_at_timestep))
            prompts_ppls = [(i[0],i[-1]) for i in txts_at_timestep]
            crt_budget = int(crt_budget/dg.branch_factors[gold_choice])
            if previous_rmse == 0:
                initial_rmse = rmse[0][-1]
                previous_rmse = rmse[0][-1]
            # print "\tbatch: %d, started at: %f, now: %f" % (b, abs(initial_rmse), abs(rmse[0][-1]))
            if (len(total_rmse) == 0):
                total_rmse = [rmse]
                total_regret = [regret]
            else:
                total_rmse.append(np.mean(rmse))
                total_regret.append(np.mean(regret))
                print(len(total_regret))
            print(best_per_bucket)
            training_examples.append([overall_sample_features,overall_rewards])
        records.append(best_per_bucket)
    file = open("surprise_bandit_epoch_e_reward_epoch"+str(epoch), 'wb')
    pickle.dump((positiveStrategy,simulator, records, total_regret), file)
    file = open("surprise_bandit_training_e_reward_epoch_"+str(epoch), 'wb')
    pickle.dump(training_examples, file)
            # (prompts, overall_sample_features, overall_rewards, best_per_bucket, anchors) = dg.generate_gpt_examples(
            #     prompts, crt_length, crt_budget=-1)
            # print("final length of prompts is ",len(prompts))








# appropriate model for reward_type='binary'
# binaryStrategy = BinaryStrategy.BinaryStrategy(arms,features)
# regret,delta = binaryStrategy.simulate(sample_features,sample_rewards,dg.W)

mean_regret = total_regret / num_experiments
mean_rmse = total_rmse / num_experiments

