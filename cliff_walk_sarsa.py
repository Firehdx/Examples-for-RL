# -*- coding:utf-8 -*-
# Train Sarsa in cliff-walking environment
import math, os, time, sys
import numpy as np
import random
import gym
from agent import SarsaAgent
##### START CODING HERE #####
# This code block is optional. You can import other libraries or define your utility functions if necessary.
from torch.utils.tensorboard import SummaryWriter
lr = 0.1
gamma = 0.9
epsilon = 1
e_decay = 0.008
writer = SummaryWriter(log_dir='./sarsaruns/lr={}-gamma={}-epsilon={}-e_decay={}'.format(lr,gamma,epsilon,e_decay))

def begin_video(episode):
    if episode >= 950:
        return True
    else:
        return False
##### END CODING HERE #####

# construct the environment
env = gym.make("CliffWalking-v0")
env = gym.wrappers.RecordVideo(env, './video', episode_trigger=begin_video)
# get the size of action space 
num_actions = env.action_space.n
all_actions = np.arange(num_actions)
# set random seed and make the result reproducible
RANDOM_SEED = 0
env.seed(RANDOM_SEED)
random.seed(RANDOM_SEED) 
np.random.seed(RANDOM_SEED) 

####### START CODING HERE #######

# construct the intelligent agent.
agent = SarsaAgent(all_actions, lr, gamma, epsilon, e_decay)

# start training
for episode in range(1000):
    # record the reward in an episode
    episode_reward = 0
    # reset env
    s = env.reset()
    # render env. You can remove all render() to turn off the GUI to accelerate training.
    #env.render()
    # choose first action
    a = agent.choose_action(s)
    # agent interacts with the environment
    for it in range(500):
        # take action
        s_, r, isdone, info = env.step(a)
        # if episode == 999:
        #     env.render()
        #env.render()
        # choose the next action
        a_ = agent.choose_action(s_)
        # update the episode reward
        episode_reward += r
        #print(f"{s} {a} {s_} {r} {isdone}")
        # agent learns from experience
        agent.learn(s, a, s_, a_, r)
        s = s_
        a = a_
        if isdone:
            time.sleep(0.1)
            break
    print('episode:', episode, 'episode_reward:', episode_reward, 'epsilon:', agent.epsilon) 
    agent.epsilon_decay()
    writer.add_scalar('episode reward', episode_reward, episode)
    writer.add_scalar('epsilon', agent.epsilon, episode)
print('\ntraining over\n')

# close the render window after training.
env.close()
writer.close()
####### END CODING HERE #######


