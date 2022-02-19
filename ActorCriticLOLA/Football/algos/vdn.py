import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import torch.multiprocessing as mp
import numpy as np
from models.mixers.vdn_net import VDNMixer
from models.mixers.qmix_net import QMixNet
from models.agents.rnn_agent import RNNAgent
from torch.optim import Adam
import pdb


class QLearner():
    def __init__(self, arg_dict, model, mixer, device=None):
        self.gamma = arg_dict["gamma"]
        self.K_epoch = arg_dict["k_epoch"]
        self.lmbda = arg_dict["lmbda"]
        self.eps_clip = arg_dict["eps_clip"]
        self.entropy_coef = arg_dict["entropy_coef"]
        self.grad_clip = arg_dict["grad_clip"]
        self.params = list(model.parameters())
        self.last_target_update_step = 0
        self.optimization_step = 0
        self.arg_dict = arg_dict

        self.n_actions = self.arg_dict["n_actions"]
        self.n_agents = self.arg_dict["n_agents"]
        self.state_shape = self.arg_dict["state_shape"]
        self.obs_shape = self.arg_dict["obs_shape"]
        input_shape = self.obs_shape
        # 根据参数决定RNN的输入维度
        if self.arg_dict["last_action"]:
            input_shape += self.n_actions
        if self.arg_dict["reuse_network"]:
            input_shape += self.n_agents

        # 神经网络
        # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        device = "cpu"
        # self.eval_rnn = RNNAgent(input_shape, self.arg_dict, device)  # 每个agent选动作的网络
        # self.target_rnn = RNNAgent(input_shape, self.arg_dict, device)

        # self.mixer = None
        # if arg_dict["mixer"] is not None:
        #     if arg_dict["mixer"] == "vdn":
        #         self.mixer = VDNMixer()
        #     elif arg_dict["mixer"] == "qmix":
        #         self.mixer = QMixNet(arg_dict)
        #     else:
        #         raise ValueError("Mixer {} not recognised".format(arg_dict["mixer"]))
        #     self.params += list(self.mixer.parameters())
        #     self.target_mixer = copy.deepcopy(self.mixer)

        # self.optimizer = Adam(params=self.params, lr=arg_dict["learning_rate"])

        self.target_model = copy.deepcopy(model)
        self.target_mixer = copy.deepcopy(mixer)
        self.eval_parameters = list(model.parameters()) + list(mixer.parameters())
        if arg_dict["optimizer"] == "RMS":
            self.optimizer = torch.optim.RMSprop(self.eval_parameters, lr=arg_dict["learning_rate"])

    def split_agents(self, value): # 输入维度：[120,32], 其中120代表2个agent的30个transition，奇数表示agent1，偶数表示agent2
        q_x_1 = torch.Tensor([])
        q_x_2 = torch.Tensor([])
        for i in range(self.arg_dict["rollout_len"]):
            q_a_1 = value[2 * i]  # (12)
            q_a_2 = value[2 * i + 1]
            q_x_1 = torch.cat([q_x_1, q_a_1], dim=0)
            q_x_2 = torch.cat([q_x_2, q_a_2], dim=0)
        return torch.stack((q_x_1, q_x_2), dim=0)  # (2, 60*32)

    def obtain_one_state(self, state): # 输入维度：[120,32,136],其中120代表2个agent的30个transition，奇数表示agent1，偶数表示agent2
        q_x_1 = torch.Tensor([])
        q_x_2 = torch.Tensor([])
        for i in range(self.arg_dict["rollout_len"]):
            q_a_1 = state[2 * i]  # (12)
            q_a_2 = state[2 * i + 1]
            q_x_1 = torch.cat([q_x_1, q_a_1], dim=0)
            q_x_2 = torch.cat([q_x_2, q_a_2], dim=0)
        return q_x_1  # (60,32,136)

    def magic_box(self, x):
        return torch.exp(x - x.detach())

    def train(self, model, mixer, data):

        ### rjq debug 0118
        # self.model.load_state_dict(model.state_dict())  # rjq 0114  传入self.model
        # if self.mixer is not None:
        #     self.mixer.load_state_dict(mixer.state_dict())  # mixer.state_dict() == self.target_mixer.state_dict()

        # model.init_hidden()
        # self.target_model.init_hidden()

        loss = []
        for mini_batch in data:
            # pdb.set_trace()
            # obs_model, h_in, actions1, avail_u, actions_onehot1, reward_agency, obs_prime, h_out, avail_u_next, done
            s, h_in, a, a1, avail_u, r, s_prime, h_out, avail_u_next, done = mini_batch

            action_prob = model.actor_forward(s)   # torch.Size([20, 4, 19])
            v = model.critic_forward(s)  # torch.Size([20, 4, 1])

            action_distribution = Categorical(action_prob)  # 20.4.19
            action = action_distribution.sample()  # 20.4
            logprobs = action_distribution.log_prob(action)  # 20.4

            logprobs = self.split_agents(logprobs) # 2, 10*4
            self_logprobs = logprobs[0]
            other_logprobs = logprobs[1]
            v = self.split_agents(v.squeeze(-1))
            values = v[1]

            gamma = 0.98

            rewards = self.split_agents(r.squeeze(-1))[0]


            # apply discount:
            cum_discount = torch.cumprod(gamma * torch.ones(*rewards.size()), dim=0) / gamma
            discounted_rewards = rewards * cum_discount
            discounted_values = values * cum_discount

            # stochastics nodes involved in rewards dependencies:
            dependencies = torch.cumsum(self_logprobs + other_logprobs, dim=0)

            # logprob of each stochastic nodes:
            stochastic_nodes = self_logprobs + other_logprobs

            # dice objective:
            dice_objective = torch.mean(torch.sum(self.magic_box(dependencies) * discounted_rewards, dim=0))

            use_baseline = True
            if use_baseline:
                # variance_reduction:
                baseline_term = torch.mean(torch.sum((1 - self.magic_box(stochastic_nodes)) * discounted_values, dim=0))
                dice_objective_ = dice_objective + baseline_term

            v_loss = torch.mean((rewards - values)**2)

            loss_all = dice_objective_ + v_loss
            loss.append(loss_all)
        # loss_ = torch.mean(loss)
        loss = torch.mean(torch.stack(loss), 0)
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.eval_parameters, self.grad_clip)
        self.optimizer.step()

        self.optimization_step += 1
        if self.optimization_step % self.arg_dict["target_update_interval"] == 0.0:
            self._update_targets(model, mixer)
            self.last_target_update_step = self.optimization_step
            print("self.last_target_update_step:---", self.last_target_update_step)

        return loss


    def _update_targets(self, model, mixer):
        self.target_model.load_state_dict(model.state_dict())
        if mixer is not None:
            self.target_mixer.load_state_dict(mixer.state_dict())

    # def cuda(self):
    #     self.model.cuda()
    #     self.target_model.cuda()
    #     if self.mixer is not None:
    #         self.mixer.cuda()
    #         self.target_mixer.cuda()

    def save_models(self, path, model, mixer):
        torch.save(model.state_dict(), "{}agent.th".format(path))
        if mixer is not None:
            torch.save(mixer.state_dict(), "{}mixer.th".format(path))
        torch.save(self.optimizer.state_dict(), "{}opt.th".format(path))
        print("Model saved :", path)

    # def load_models(self, path):
    #     self.model.load_models(path)
    #     self.target_model.load_models(path)
    #     if self.mixer is not None:
    #         self.mixer.load_state_dict(torch.load("{}/mixer.th".format(path)),
    #                                    map_location=lambda storage, loc: storage)
    #     self.optimizer.load_state_dict(torch.load("{}/opt.th".format(path)), map_location=lambda storage, loc: storage)
