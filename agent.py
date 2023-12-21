# -*- coding:utf-8 -*-
import math, os, time, sys
import numpy as np
import gym
##### START CODING HERE #####
# This code block is optional. You can import other libraries or define your utility functions if necessary.
def e_decay(epsilon, e_decay):
    return epsilon / (1 + e_decay)

def epsilon_greedy(epsilon, all_actions, max_action):
    p = np.random.random()
    if p < epsilon:
        return np.random.choice(all_actions)
    else:
        return max_action

##### END CODING HERE #####

# ------------------------------------------------------------------------------------------- #

class SarsaAgent(object):
    ##### START CODING HERE #####
    def __init__(self, all_actions, lr, gamma, epsilon, e_decay):
        """initialize the agent. Maybe more function inputs are needed."""
        self.all_actions = all_actions
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.e_decay = e_decay
        #Qlist is a dict: {s1:np.array([q0,q1,q2,q3]), s2:np.array([q0,q1,q2,q3]),,,}
        self.Qlist = dict()

    def choose_action(self, observation):
        """choose action with epsilon-greedy algorithm."""
        #action = np.random.choice(self.all_actions)
        Qs = self.Qlist.get(observation)
        if Qs is None:
            Qs = np.array([0.0, 0.0, 0.0, 0.0])
            self.Qlist[observation] = Qs
        max_action = self.all_actions[np.argmax(Qs)]
        action = epsilon_greedy(self.epsilon, self.all_actions, max_action)
        return action
    
    def learn(self, last_state, last_action, now_state, now_action, reward):
        """learn from experience"""
        self.Qlist[last_state][last_action] += self.lr*(reward+self.gamma*self.Qlist[now_state][now_action]
                                                        -self.Qlist[last_state][last_action])

    def epsilon_decay(self):
        self.epsilon = e_decay(self.epsilon, self.e_decay)
    ##### END CODING HERE #####


class QLearningAgent(object):
    ##### START CODING HERE #####
    def __init__(self, all_actions, lr, gamma, epsilon, e_decay):
        """initialize the agent. Maybe more function inputs are needed."""
        self.all_actions = all_actions
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.e_decay = e_decay
        self.Qlist = dict()

    def choose_action(self, observation):
        """choose action with epsilon-greedy algorithm."""
        #action = np.random.choice(self.all_actions)
        Qs = self.Qlist.get(observation)
        if Qs is None:
            Qs = np.array([0.0, 0.0, 0.0, 0.0])
            self.Qlist[observation] = Qs
        max_action = self.all_actions[np.argmax(Qs)]
        action = epsilon_greedy(self.epsilon, self.all_actions, max_action)
        return action
    
    def learn(self, last_state, last_action, now_state, reward):
        """learn from experience"""
        Qs_now = self.Qlist.get(now_state)
        if Qs_now is None:
            Qs_now = np.array([0.0, 0.0, 0.0, 0.0])
            self.Qlist[now_state] = Qs_now
        self.Qlist[last_state][last_action] += self.lr*(reward+self.gamma*max(Qs_now)
                                                        -self.Qlist[last_state][last_action])

    def epsilon_decay(self):
        self.epsilon = e_decay(self.epsilon, self.e_decay)
    ##### END CODING HERE #####
