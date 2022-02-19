import torch
import numpy as np
import sys
import gym
from torch.autograd import Variable

sys.path.append("./")
from base_net import *
from torch import nn, optim
import torch.nn.functional as F
from torch.distributions import Categorical


class ActorCritic(nn.Module):
    def __init__(self, args):
        super(ActorCritic, self).__init__()
        self.input_size, self.output_size, self.device, self.actor_lr, self.critic_lr, self.len_rollout = args
        self.actor = Policy_net(args=(self.input_size, self.output_size))
        self.critic = Q_net(args=(self.input_size, 1))

        # self.buffer = ReplayBuffer(args=(10000))
        # self.optimizer_actor = optim.Adam(list(self.actor.parameters())+list(self.critic.parameters()), lr=self.actor_lr)
        self.optimizer_actor = optim.Adam(self.actor.parameters(), lr=self.actor_lr)
        self.optimizer_critic = optim.Adam(self.critic.parameters(), lr=self.critic_lr)

    def get_policy(self, inputs):
        return F.softmax(self.actor(inputs))

    def save_trans(self, transition):
        self.buffer.save_trans(transition)

    # def to_tensor(self, items):
    #     s, a, r, s_next, done = items
    #     s = torch.FloatTensor(s).to(self.device)
    #     a = torch.LongTensor(a).to(self.device)
    #     r = torch.FloatTensor(r).to(self.device)
    #     s_next = torch.FloatTensor(s_next).to(self.device)
    #     done = torch.FloatTensor(done).to(self.device)
    #
    #     return s, a.unsqueeze(-1), r.unsqueeze(-1), s_next, done.unsqueeze(-1)


    def to_one_hot(self, action):
        one_hot = torch.zeros(1, 5).long()
        one_hot_ = one_hot.scatter_(dim=1,index=torch.tensor([[action]]),src=torch.ones(1, 5).long()).squeeze(0)
        return one_hot_

    def select_action(self, inputs):
        action_prob = self.get_policy(inputs)
        action_distribution = Categorical(action_prob)
        action = action_distribution.sample()
        logprobs = action_distribution.log_prob(action)

        return action.item(), logprobs


    def in_lookahead(self, env, agent):
        # device = 'cuda' if torch.cuda.is_available() else 'cpu'
        device = 'cpu'
        s = env.reset()
        # buffer = ReplayBuffer2(args=(100))
        other_memory = Memory()
        for t in range(self.len_rollout):
            s = torch.FloatTensor(s).to(device)
            action1, log_p1 = self.select_action(s[0])
            action1_one_hot = self.to_one_hot(action1)
            value1 = self.critic(s[0])
            action2, log_p2 = agent.select_action(s[1])
            action2_one_hot = self.to_one_hot(action2)
            value2 = agent.critic(s[1])

            p_actions_onehot = [action1_one_hot, action2_one_hot]

            s_next, reward, done, info = env.step(p_actions_onehot)

            reward = torch.tensor(reward[0])

            other_memory.add(log_p2, log_p1, value2, reward)

            # transition = (log_p2, log_p1, value2, reward)  # torch.from_numpy(reward).float()
            # buffer.save_trans(transition)
            s = s_next
            if done:
                break

        other_objective = other_memory.dice_objective()
        other_objective.backward(retain_graph=True)
        grads = [p.grad for p in agent.actor.parameters()]
        # grad = torch.autograd.grad(other_objective, (agent.actor), create_graph=True)[0]
        
        return grads

    def out_lookahead(self, env, agent):
        # device = 'cuda' if torch.cuda.is_available() else 'cpu'
        device = 'cpu'
        s = env.reset()
        self.buffer = Memory()
        for t in range(self.len_rollout):
            s = torch.FloatTensor(s).to(device)
            action1, log_p1 = self.select_action(s[0])
            action1_one_hot = self.to_one_hot(action1)
            value1 = self.critic(s[0])
            action2, log_p2 = agent.select_action(s[1])
            action2_one_hot = self.to_one_hot(action2)
            value2 = agent.critic(s[1])

            p_actions_onehot = [action1_one_hot, action2_one_hot]

            s_next, reward, done, info = env.step(p_actions_onehot)

            reward = torch.tensor(reward[0])

            self.buffer.add(log_p1, log_p2, value1, reward)

            # transition = (log_p1, log_p2, value1, torch.from_numpy(reward).float())
            # self.buffer.save_trans(transition)
            s = s_next
            if done:
                break

        objective = self.buffer.dice_objective()

        self.optimizer_actor.zero_grad()
        objective.backward(retain_graph=True)
        self.optimizer_actor.step()

        v_loss = self.buffer.value_loss()

        self.optimizer_critic.zero_grad()
        v_loss.backward()
        self.optimizer_critic.step()


        return objective, v_loss







