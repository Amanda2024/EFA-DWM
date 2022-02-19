import numpy as np
import torch
from policy.vdn import VDN
from policy.qmix import QMIX
from policy.coma import COMA
from policy.reinforce import Reinforce
from policy.central_v import CentralV
from policy.qtran_alt import QtranAlt
from policy.qtran_base import QtranBase
from policy.maven import MAVEN
from policy.ssg_q import SSG_Q
from policy.ssg_vdn import SSG_VDN
from policy.ssg_vdn_two_step import SSG_VDN_TWO
from torch.distributions import Categorical
from policy.asg_vdn_new import ASG_VDN

# Agent no communication
class Agents:
    def __init__(self, args):
        self.n_actions = args.n_actions
        self.n_agents = args.n_agents
        self.state_shape = args.state_shape
        self.obs_shape = args.obs_shape
        if args.alg == 'vdn':
            self.policy = VDN(args)
        elif args.alg == 'qmix':
            self.policy = QMIX(args)
        elif args.alg == 'coma':
            self.policy = COMA(args)
        elif args.alg == 'qtran_alt':
            self.policy = QtranAlt(args)
        elif args.alg == 'qtran_base':
            self.policy = QtranBase(args)
        elif args.alg == 'maven':
            self.policy = MAVEN(args)
        elif args.alg == 'central_v':
            self.policy = CentralV(args)
        elif args.alg == 'reinforce':
            self.policy = Reinforce(args)
        elif args.alg == 'ssg':
            self.policy = SSG_Q(args)
        elif args.alg == 'ssg_vdn':
            self.policy = SSG_VDN(args)
        elif args.alg == 'ssg_vdn_two':
            self.policy = SSG_VDN_TWO(args)
        elif args.alg == 'asg_vdn':
            self.policy = ASG_VDN(args)
        else:
            raise Exception("No such algorithm")
        self.args = args
        print('Init Agents')


    def choose_action_asg(self, obs, last_action, avail_actions, epsilon, step, leader_id_one_hot, maven_z=None, evaluate=False):

        inputs = obs.copy()
        inputs = torch.tensor(inputs, dtype=torch.float32)
        if step % self.args.maintain_step == 0:
            leader_id_one_hot, self.policy.election_hidden = self.policy.get_leader_id(inputs, self.policy.election_hidden)
        leader_id = torch.argmax(leader_id_one_hot.clone().detach()).item()

        folower_indices = [j for j in range(self.args.n_agents)]
        folower_indices.remove(leader_id)
        folower_indices = torch.tensor(np.array(folower_indices), dtype=torch.int64)
        inputs = inputs.numpy()

        avail_actions_ind_l = np.nonzero(avail_actions[leader_id])[0]  # index of actions which can be choose
        avail_actions_ind_f = [np.nonzero(avail_actions[j])[0] for j in
                               folower_indices]  # index of actions which can be choose

        if self.args.scenario == 'simple_spread.py':
            tmp_inputs = [[]] * self.args.n_agents
            for i in range(len(tmp_inputs)):
                tmp_inputs[i] = inputs[i][np.newaxis, :]  # inputs: n_agents x 1 x obs_shape
            inputs = tmp_inputs
        # transform agent_num to onehot vector
        # agent_id = np.zeros(self.n_agents)
        # agent_id[agent_num] = 1.
        agent_id = np.eye(self.n_agents)

        if self.args.last_action:
            inputs = np.array([np.hstack((inputs[i].reshape([1, -1]), last_action[i].reshape([1, -1]))) for i in
                      range(self.n_agents)])
            # inputs_l = np.hstack((inputs.reshape([1, -1]), last_action.reshape([1, -1])))  # 1 x obs_shape+act_dim
            # inputs_f = np.array([np.hstack(
            #     (inputs[j].reshape([1, -1]), last_action[j].reshape([1, -1]))) for j in
            #     folower_indices])  # 2 x 1 x obs_shape+act_dim
        if self.args.reuse_network:
            inputs = np.array([np.hstack((inputs[i].reshape([1, -1]), agent_id[i].reshape([1, -1]))) for i in
                      range(self.n_agents)])
            # inputs_l = np.hstack((inputs_l, agent_id[leader_id].reshape([1, -1])))[:, np.newaxis,
            #            :]  # 1 x 1 x input_shape
            # inputs_f = np.array(
            #     [np.hstack((inputs_f[j], agent_id[j].reshape([1, -1]))) for j in range(self.args.n_agents - 1)])

        inputs_l = torch.mm(leader_id_one_hot.double(), torch.tensor(inputs).squeeze(1))
        # inputs_l = inputs_l.detach().numpy()
        follower_id_one_hot = torch.ones_like(leader_id_one_hot) - leader_id_one_hot
        # inputs_f = torch.mm(follower_id_one_hot.double(), torch.tensor(inputs).squeeze(1))
        inputs_f = np.array([inputs[j] for j in folower_indices])
        inputs_f = torch.tensor(inputs_f, dtype=torch.float32)

        election_hidden_state = self.policy.election_hidden
        hidden_state = self.policy.eval_hidden
        hidden_state_follower = self.policy.eval_follower_hidden

        # transform the shape of inputs from (42,) to (1,42)
        # inputs = torch.tensor(inputs, dtype=torch.float32).unsqueeze(0)
        inputs_l = torch.tensor(inputs_l, dtype=torch.float32)
        inputs_f = torch.tensor(inputs_f, dtype=torch.float32)
        avail_actions = torch.tensor(avail_actions, dtype=torch.float32).unsqueeze(0)
        if self.args.cuda:
            # inputs = inputs.cuda()
            inputs_l = inputs_l.cuda()
            inputs_f = inputs_f.cuda()
            election_hidden_state = election_hidden_state.cuda()
            hidden_state = hidden_state.cuda()
            hidden_state_follower = hidden_state_follower.cuda()

        # get q value
        q_value_l, self.policy.eval_hidden = self.policy.eval_rnn(inputs_l.unsqueeze(1), hidden_state)
        q_value_f, self.policy.eval_follower_hidden = self.policy.eval_follower(inputs_f, q_value_l,
                                                                                hidden_state_follower)
        q_value_l = q_value_l.squeeze(1)
        q_value_f = q_value_f.squeeze(1)

        # choose action from q value
        if self.args.alg == 'coma' or self.args.alg == 'central_v' or self.args.alg == 'reinforce':
            action = self._choose_action_from_softmax(q_value.cpu(), avail_actions, epsilon, evaluate)
        elif self.args.alg == 'ssg' or self.args.alg == 'ssg_vdn' or self.args.alg == 'ssg_vdn_two':
            q_value_l[avail_actions[:, 0, :] == 0.0] = -float("inf")
            q_value_f[avail_actions[:, 1, :] == 0.0] = -float("inf")
            if np.random.uniform() < epsilon:
                action_l = np.random.choice(avail_actions_ind_l)
                action_f = np.random.choice(avail_actions_ind_f)
            else:
                action_l = torch.argmax(q_value_l)  # The first agent is leader
                action_f = torch.argmax(q_value_f)  # The first agent is leader
        elif self.args.alg == 'asg_vdn':
            q_value_l[avail_actions[:, leader_id, :] == 0.0] = -float("inf")
            q_value_f = q_value_f.unsqueeze(1)  # n_agetns-1 x act_dims
            for x in range(q_value_f.shape[0]):
                q_value_f[x][avail_actions[:, folower_indices[x], :] == 0.0] = -float("inf")
            q_value_f = q_value_f.squeeze(1)
            if np.random.uniform() < epsilon:
                action_l = np.random.choice(avail_actions_ind_l)
                action_f = [np.random.choice(avail_actions_ind_f[j]) for j in range(self.args.n_agents - 1)]
            else:
                action_l = torch.argmax(q_value_l)  # The leader is automatic figured
                action_f = torch.argmax(q_value_f, dim=-1)
            final_act_f = list(action_f)
            # final_act_f.insert(leader_id, action_l)
        else:
            q_value[avail_actions == 0.0] = - float("inf")
            if np.random.uniform() < epsilon:
                action = np.random.choice(avail_actions_ind)  # action是一个整数
            else:
                action = torch.argmax(q_value)

        actions_ = [0, 0, 0, 0, 0]
        actions_[leader_id] = action_l
        actions_[folower_indices[0]] = final_act_f[0]
        actions_[folower_indices[1]] = final_act_f[1]
        actions_[folower_indices[2]] = final_act_f[2]
        actions_[folower_indices[3]] = final_act_f[3]
        return actions_, leader_id_one_hot
        # return action_l if agent_num == leader_id else final_act_f[agent_num]

    def choose_action_ssg(self, obs, last_action, agent_num, avail_actions, epsilon, maven_z=None, evaluate=False):
        inputs = obs.copy()
        avail_actions_ind_l = np.nonzero(avail_actions[0])[0]  # index of actions which can be choose
        avail_actions_ind_f = np.nonzero(avail_actions[1])[0]  # index of actions which can be choose

        if self.args.scenario == 'simple_spread.py':
            for i in range(len(inputs)):
                inputs[i] = inputs[i][np.newaxis, :]

        # transform agent_num to onehot vector
        # agent_id = np.zeros(self.n_agents)
        # agent_id[agent_num] = 1.
        agent_id = np.eye(self.n_agents)

        if self.args.last_action:
            inputs_l = np.hstack((inputs[0], last_action[0].reshape([inputs[0].shape[0], -1])))
            inputs_f = np.hstack((inputs[1], last_action[1].reshape([inputs[1].shape[0], -1])))
        if self.args.reuse_network:
            inputs_l = np.hstack((inputs_l, agent_id[0].reshape([inputs[0].shape[0], -1])))
            inputs_f = np.hstack((inputs_f, agent_id[1].reshape([inputs[1].shape[0], -1])))
        hidden_state = self.policy.eval_hidden
        hidden_state_follower = self.policy.eval_follower_hidden

        # transform the shape of inputs from (42,) to (1,42)
        # inputs = torch.tensor(inputs, dtype=torch.float32).unsqueeze(0)
        inputs_l = torch.tensor(inputs_l, dtype=torch.float32)
        inputs_f = torch.tensor(inputs_f, dtype=torch.float32)
        avail_actions = torch.tensor(avail_actions, dtype=torch.float32).unsqueeze(0)
        if self.args.cuda:
            # inputs = inputs.cuda()
            inputs_l = inputs_l.cuda()
            inputs_f = inputs_f.cuda()
            hidden_state = hidden_state.cuda()

        # get q value
        q_value_l, self.policy.eval_hidden = self.policy.eval_rnn(inputs_l, hidden_state)
        q_value_f, self.policy.eval_follower_hidden = self.policy.eval_follower(inputs_f, q_value_l, hidden_state_follower)

        # get mixing value
        # q_value_mix = self.policy.eval_qmix_net(torch.stack((q_value_l, q_value_f), dim=1), torch.from_numpy(inputs).permute([1,2,0]))
        # q_value_mix = self.policy.eval_qmix_net(q_value, inputs)

        # choose action from q value
        if self.args.alg == 'coma' or self.args.alg == 'central_v' or self.args.alg == 'reinforce':
            action = self._choose_action_from_softmax(q_value.cpu(), avail_actions, epsilon, evaluate)
        elif self.args.alg == 'ssg' or self.args.alg == 'ssg_vdn' or self.args.alg == 'ssg_vdn_two':
            q_value_l[avail_actions[:,0,:] == 0.0] = -float("inf")
            q_value_f[avail_actions[:,1,:] == 0.0] = -float("inf")
            if np.random.uniform() < epsilon:
                action_l = np.random.choice(avail_actions_ind_l)
                action_f = np.random.choice(avail_actions_ind_f)
            else:
                action_l = torch.argmax(q_value_l)  # The first agent is leader
                action_f = torch.argmax(q_value_f)  # The first agent is leader
        else:
            q_value[avail_actions == 0.0] = - float("inf")
            if np.random.uniform() < epsilon:
                action = np.random.choice(avail_actions_ind)  # action是一个整数
            else:
                action = torch.argmax(q_value)
        return action_l if agent_num ==0 else action_f

    def choose_action(self, obs, last_action, agent_num, avail_actions, epsilon, maven_z=None, evaluate=False):
        inputs = obs.copy()
        avail_actions_ind = np.nonzero(avail_actions)[0]  # index of actions which can be choose

        # transform agent_num to onehot vector
        agent_id = np.zeros(self.n_agents)
        agent_id[agent_num] = 1.

        if self.args.last_action:
            inputs = np.hstack((inputs, last_action.reshape([inputs.shape[0], -1])))
        if self.args.reuse_network:
            inputs = np.hstack((inputs, agent_id.reshape([inputs.shape[0], -1])))
        hidden_state = self.policy.eval_hidden[:, agent_num, :] if self.args.alg != 'ssg' else self.policy.eval_hidden
        if self.args.alg == 'ssg' or self.args.alg == 'ssg_vdn' or self.args.alg == 'ssg_vdn_two':
            hidden_state_follower = self.policy.eval_follower_hidden

        # transform the shape of inputs from (42,) to (1,42)
        # inputs = torch.tensor(inputs, dtype=torch.float32).unsqueeze(0)
        inputs = torch.tensor(inputs, dtype=torch.float32)
        avail_actions = torch.tensor(avail_actions, dtype=torch.float32).unsqueeze(0)
        if self.args.cuda:
            inputs = inputs.cuda()
            hidden_state = hidden_state.cuda()

        # get q value
        if self.args.alg == 'maven':
            maven_z = torch.tensor(maven_z, dtype=torch.float32).unsqueeze(0)
            if self.args.cuda:
                maven_z = maven_z.cuda()
            q_value, self.policy.eval_hidden[:, agent_num, :] = self.policy.eval_rnn(inputs, hidden_state, maven_z)
        elif self.args.alg == 'ssg' or self.args.alg == 'ssg_vdn' or self.args.alg == 'ssg_vdn_two': #
            if agent_num == 0:
                q_value, self.policy.eval_hidden = self.policy.eval_rnn(inputs, hidden_state)
            else:
                q_value, self.policy.eval_follower_hidden = self.policy.eval_follower(inputs, q_value, hidden_state_follower)
            # q_value_mix = self.policy.eval_qmix_net(q_value, inputs)
        else:
            q_value, self.policy.eval_hidden[:, agent_num, :] = self.policy.eval_rnn(inputs, hidden_state)

        # choose action from q value
        if self.args.alg == 'coma' or self.args.alg == 'central_v' or self.args.alg == 'reinforce':
            action = self._choose_action_from_softmax(q_value.cpu(), avail_actions, epsilon, evaluate)
        elif self.args.alg == 'ssg' or self.args.alg == 'ssg_vdn' or self.args.alg == 'ssg_vdn_two':
            q_value[avail_actions == 0.0] = -float("inf")
            if np.random.uniform() < epsilon:
                action = np.random.choice(avail_actions_ind)
            else:
                action = torch.argmax(q_value) # The first agent is leader
        else:
            q_value[avail_actions == 0.0] = - float("inf")
            if np.random.uniform() < epsilon:
                action = np.random.choice(avail_actions_ind)  # action是一个整数
            else:
                action = torch.argmax(q_value)
        return action



    def _choose_action_from_softmax(self, inputs, avail_actions, epsilon, evaluate=False):
        """
        :param inputs: # q_value of all actions
        """
        action_num = avail_actions.sum(dim=1, keepdim=True).float().repeat(1, avail_actions.shape[
            -1])  # num of avail_actions
        # 先将Actor网络的输出通过softmax转换成概率分布
        prob = torch.nn.functional.softmax(inputs, dim=-1)
        # add noise of epsilon
        prob = ((1 - epsilon) * prob + torch.ones_like(prob) * epsilon / action_num)
        prob[avail_actions == 0] = 0.0  # 不能执行的动作概率为0

        """
        不能执行的动作概率为0之后，prob中的概率和不为1，这里不需要进行正则化，因为torch.distributions.Categorical
        会将其进行正则化。要注意在训练的过程中没有用到Categorical，所以训练时取执行的动作对应的概率需要再正则化。
        """

        if epsilon == 0 and evaluate:
            action = torch.argmax(prob)
        else:
            action = Categorical(prob).sample().long()
        return action

    def _get_max_episode_len(self, batch):
        terminated = batch['terminated']
        episode_num = terminated.shape[0]
        max_episode_len = 0
        for episode_idx in range(episode_num):
            for transition_idx in range(self.args.episode_limit):
                if terminated[episode_idx, transition_idx, 0] == 1:
                    if transition_idx + 1 >= max_episode_len:
                        max_episode_len = transition_idx + 1
                    break
        return max_episode_len

    def train(self, batch, train_step, epsilon=None):  # coma needs epsilon for training

        # different episode has different length, so we need to get max length of the batch
        max_episode_len = self._get_max_episode_len(batch)
        for key in batch.keys():
            if key != 'z':
                batch[key] = batch[key][:, :max_episode_len]
        updated_loss = self.policy.learn(batch, max_episode_len, train_step, epsilon)
        if train_step > 0 and train_step % self.args.save_cycle == 0:
            self.policy.save_model(train_step)

        return updated_loss  # log the loss after updating

# Agent for communication
class CommAgents:
    def __init__(self, args):
        self.n_actions = args.n_actions
        self.n_agents = args.n_agents
        self.state_shape = args.state_shape
        self.obs_shape = args.obs_shape
        alg = args.alg
        if alg.find('reinforce') > -1:
            self.policy = Reinforce(args)
        elif alg.find('coma') > -1:
            self.policy = COMA(args)
        elif alg.find('central_v') > -1:
            self.policy = CentralV(args)
        else:
            raise Exception("No such algorithm")
        self.args = args
        print('Init CommAgents')

    # 根据weights得到概率，然后再根据epsilon选动作
    def choose_action(self, weights, avail_actions, epsilon, evaluate=False):
        weights = weights.unsqueeze(0)
        avail_actions = torch.tensor(avail_actions, dtype=torch.float32).unsqueeze(0)
        action_num = avail_actions.sum(dim=1, keepdim=True).float().repeat(1, avail_actions.shape[-1])  # 可以选择的动作的个数
        # 先将Actor网络的输出通过softmax转换成概率分布
        prob = torch.nn.functional.softmax(weights, dim=-1)
        # 在训练的时候给概率分布添加噪音
        prob = ((1 - epsilon) * prob + torch.ones_like(prob) * epsilon / action_num)
        prob[avail_actions == 0] = 0.0  # 不能执行的动作概率为0

        """
        不能执行的动作概率为0之后，prob中的概率和不为1，这里不需要进行正则化，因为torch.distributions.Categorical
        会将其进行正则化。要注意在训练的过程中没有用到Categorical，所以训练时取执行的动作对应的概率需要再正则化。
        """

        if epsilon == 0 and evaluate:
            # 测试时直接选最大的
            action = torch.argmax(prob)
        else:
            action = Categorical(prob).sample().long()
        return action

    def get_action_weights(self, obs, last_action):
        obs = torch.tensor(obs, dtype=torch.float32)
        last_action = torch.tensor(last_action, dtype=torch.float32)
        inputs = list()
        inputs.append(obs)
        # 给obs添加上一个动作、agent编号
        if self.args.last_action:
            inputs.append(last_action)
        if self.args.reuse_network:
            inputs.append(torch.eye(self.args.n_agents))
        inputs = torch.cat([x for x in inputs], dim=1)
        if self.args.cuda:
            inputs = inputs.cuda()
            self.policy.eval_hidden = self.policy.eval_hidden.cuda()
        weights, self.policy.eval_hidden = self.policy.eval_rnn(inputs, self.policy.eval_hidden)
        weights = weights.reshape(self.args.n_agents, self.args.n_actions)
        return weights.cpu()

    def _get_max_episode_len(self, batch):
        terminated = batch['terminated']
        episode_num = terminated.shape[0]
        max_episode_len = 0
        for episode_idx in range(episode_num):
            for transition_idx in range(self.args.episode_limit):
                if terminated[episode_idx, transition_idx, 0] == 1:
                    if transition_idx + 1 >= max_episode_len:
                        max_episode_len = transition_idx + 1
                    break
        return max_episode_len

    def train(self, batch, train_step, epsilon=None):  # coma在训练时也需要epsilon计算动作的执行概率
        # 每次学习时，各个episode的长度不一样，因此取其中最长的episode作为所有episode的长度
        max_episode_len = self._get_max_episode_len(batch)
        for key in batch.keys():
            batch[key] = batch[key][:, :max_episode_len]
        updated_loss = self.policy.learn(batch, max_episode_len, train_step, epsilon)
        if train_step > 0 and train_step % self.args.save_cycle == 0:
            self.policy.save_model(train_step)

