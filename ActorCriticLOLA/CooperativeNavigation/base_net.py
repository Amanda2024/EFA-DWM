import torch
import numpy as np
import random
import collections
import torch.nn.functional as F

from collections import deque
from torch import nn, optim

embedding_size = 64
rnn_embedding_size = 32

'''
name: q value net
description: 
'''


class Q_net(nn.Module):
    def __init__(self, args):
        super(Q_net, self).__init__()
        self.input_size, self.output_size = args
        self.q_net = nn.Sequential(
            nn.Linear(self.input_size, embedding_size),
            nn.ReLU(),
            nn.Linear(embedding_size, self.output_size)
        )

    def forward(self, inputs):
        return self.q_net(inputs)

'''
name: policy net
description: 
'''
class Policy_net(nn.Module):
    def __init__(self, args):
        super(Policy_net, self).__init__()
        self.input_size, self.output_size = args
        self.actor = nn.Sequential(
            nn.Linear(self.input_size, embedding_size),
            nn.ReLU(),
            nn.Linear(embedding_size, self.output_size)
        )

    def forward(self, inputs):
        return self.actor(inputs)

def magic_box(x):
    return torch.exp(x - x.detach())

class Memory():
    def __init__(self):
        self.self_logprobs = []
        self.other_logprobs = []
        self.values = []
        self.rewards = []

    def add(self, lp, other_lp, v, r):
        self.self_logprobs.append(lp)
        self.other_logprobs.append(other_lp)
        self.values.append(v)
        self.rewards.append(r)

    def dice_objective(self):
        gamma = 0.98

        self_logprobs = torch.stack(self.self_logprobs, dim=0)   # 128.150
        other_logprobs = torch.stack(self.other_logprobs, dim=0) # 128.150
        values = torch.stack(self.values, dim=0)                 # 128.150
        rewards = torch.stack(self.rewards, dim=0)               # 128.150

        # apply discount:
        cum_discount = torch.cumprod(gamma * torch.ones(*rewards.size()), dim=0)/gamma
        discounted_rewards = rewards * cum_discount
        discounted_values = values * cum_discount

        # stochastics nodes involved in rewards dependencies:
        dependencies = torch.cumsum(self_logprobs + other_logprobs, dim=0)

        # logprob of each stochastic nodes:
        stochastic_nodes = self_logprobs + other_logprobs

        # dice objective:
        dice_objective = torch.mean(torch.sum(magic_box(dependencies) * discounted_rewards, dim=0))

        use_baseline = True
        if use_baseline:
            # variance_reduction:
            baseline_term = torch.mean(torch.sum((1 - magic_box(stochastic_nodes)) * discounted_values, dim=0))
            dice_objective_ = dice_objective + baseline_term

        return - dice_objective_ # want to minimize -objective

    def value_loss(self):
        values = torch.stack(self.values, dim=0)
        rewards = torch.stack(self.rewards, dim=0)
        return torch.mean((rewards - values)**2)