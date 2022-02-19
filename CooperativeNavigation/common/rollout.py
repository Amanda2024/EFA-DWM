import numpy as np
import torch
from torch.distributions import one_hot_categorical
import time
import copy


class RolloutWorker:
    def __init__(self, env, agents, args):
        self.env = env
        self.agents = agents
        self.episode_limit = args.episode_limit
        self.n_actions = args.n_actions
        self.n_agents = args.n_agents
        self.state_shape = args.state_shape
        self.obs_shape = args.obs_shape
        self.args = args

        self.epsilon = args.epsilon
        self.anneal_epsilon = args.anneal_epsilon
        self.min_epsilon = args.min_epsilon
        print('Init RolloutWorker')

    def generate_episode(self, episode_num=None, evaluate=False):
        if self.args.replay_dir != '' and evaluate and episode_num == 0:  # prepare for save replay of evaluation
            self.env.close()
        o, u, r, s, avail_u, u_onehot, terminate, padded, lead_ids = [], [], [], [], [], [], [], [], []
        self.env.reset()
        terminated = False
        win_tag = False
        step = 0
        # episode_reward = 0  # cumulative rewards
        episode_reward = [0] * self.args.n_agents  # cumulative rewards for two agents
        last_action = np.zeros((self.args.n_agents, self.args.n_actions))
        self.agents.policy.init_hidden(1)

        # epsilon
        epsilon = 0 if evaluate else self.epsilon
        if self.args.epsilon_anneal_scale == 'episode':
            epsilon = epsilon - self.anneal_epsilon if epsilon > self.min_epsilon else epsilon
        if self.args.epsilon_anneal_scale == 'epoch':
            if episode_num == 0:
                epsilon = epsilon - self.anneal_epsilon if epsilon > self.min_epsilon else epsilon

        # sample z for maven
        if self.args.alg == 'maven':
            state = self.env.get_state()
            state = torch.tensor(state, dtype=torch.float32)
            if self.args.cuda:
                state = state.cuda()
            z_prob = self.agents.policy.z_policy(state)
            maven_z = one_hot_categorical.OneHotCategorical(z_prob).sample()
            maven_z = list(maven_z.cpu())

        leader_id_one_hot = torch.zeros(self.n_agents)
        while not terminated and step < self.episode_limit:
            # time.sleep(0.2)
            obs = self.env.get_obs() # for particle env
            # obs = self.env.get_obs()
            state = obs
            actions, avail_actions, actions_onehot = [], [], []
            # for agent_id in range(self.n_agents):
            #     # avail_action = self.env.get_avail_agent_actions(agent_id)
            #     avail_action = [1 for i in range(self.args.n_actions)] # for particle env
            #     if self.args.alg == 'maven':
            #         action = self.agents.choose_action(obs[agent_id], last_action[agent_id], agent_id,
            #                                            avail_action, epsilon, maven_z, evaluate)
            #     elif self.args.alg == 'ssg' or self.args.alg == 'ssg_vdn' or self.args.alg == 'ssg_vdn_two': # Put both observation into the neural networks
            #         avail_action_s = [[1 for i in range(self.args.n_actions)] for j in range(self.n_agents)]
            #         action = self.agents.choose_action_ssg(obs, last_action, agent_id,
            #                                            avail_action_s, epsilon, evaluate)
            #     elif self.args.alg == 'asg_vdn':
            #         avail_action_s = [[1 for i in range(self.args.n_actions)] for j in range(self.n_agents)]
            #         action = self.agents.choose_action_asg(obs, last_action, agent_id,
            #                                                avail_action_s, epsilon, evaluate)
            #     else:
            #         action = self.agents.choose_action(obs[agent_id], last_action[agent_id], agent_id,
            #                                            avail_action, epsilon, evaluate)
            #     # generate onehot vector of th action
            #     action_onehot = np.zeros(self.args.n_actions)
            #     action_onehot[action] = 1
            #     actions.append(action)
            #     actions_onehot.append([list(action_onehot)])
            #     avail_actions.append(avail_action)
            #     last_action[agent_id] = action_onehot

            if self.args.alg == 'asg_vdn' or self.args.alg == 'asg_vdn_two':
                avail_action_s = [[1 for i in range(self.args.n_actions)] for j in range(self.n_agents)]
                actions_, leader_id_one_hot = self.agents.choose_action_asg(obs, last_action,
                                                       avail_action_s, epsilon, step, leader_id_one_hot, evaluate)
                leader_id = torch.argmax(leader_id_one_hot.clone().detach()).item()
                # print(leader_id)
            ## generate onehot vector of th action
            avail_action = [1 for i in range(self.args.n_actions)]  # for particle env
            for agent_id in range(self.n_agents):
                action = actions_[agent_id]
                action_onehot = np.zeros(self.args.n_actions)
                action_onehot[action] = 1
                actions.append(action)
                actions_onehot.append([list(action_onehot)])
                avail_actions.append(avail_action)
                last_action[agent_id] = action_onehot

            if self.args.scenario == 'simple_spread.py':
                p_actions_onehot = [actions_onehot[i][0] for i in range(len(actions_onehot))]  # 3-d --> 2-d
                _, reward, terminated, info = self.env.step(p_actions_onehot)
                terminated = terminated[0] + 0
            else:
                _, reward, terminated, info = self.env.step(actions_onehot, step)
            reward_agency = copy.deepcopy(reward) # for reuse and change the value next timestep
            terminated_agency = copy.deepcopy(terminated)
            win_tag = True if terminated and 'battle_won' in info and info['battle_won'] else False
            if self.args.scenario == 'simple_spread.py':
                obs, state = np.array(obs), np.array(state)
            o.append(obs.reshape([self.n_agents, self.obs_shape]))
            s.append(state.reshape([self.n_agents, self.obs_shape]))
            u.append(np.reshape(actions, [self.n_agents, 1]))
            u_onehot.append(np.array(actions_onehot).reshape([self.n_agents, self.n_actions]))
            avail_u.append(np.array(avail_actions))
            r.append([reward_agency])
            terminate.append([terminated_agency])
            padded.append([0.])
            lead_ids.append([leader_id])
            for agent in range(len(reward)):
                episode_reward[agent] += reward[agent]
            step += 1
            if self.args.epsilon_anneal_scale == 'step':
                epsilon = epsilon - self.anneal_epsilon if epsilon > self.min_epsilon else epsilon
        # last obs
        if self.args.scenario == 'simple_spread.py' and len(terminate) == self.args.episode_limit:
            terminate[-1] = [1]

        o.append(obs.reshape([self.n_agents, self.obs_shape]))
        s.append(state.reshape([self.n_agents, self.obs_shape]))
        o_next = o[1:]
        s_next = s[1:]
        o = o[:-1]
        s = s[:-1]
        # get avail_action for last obs，because target_q needs avail_action in training
        avail_actions = []
        for agent_id in range(self.n_agents):
            if self.args.scenario == 'simple_spread.py':
                avail_action = [1 for i in range(self.args.n_actions)]
            else:
                avail_action = self.env.get_avail_agent_actions(agent_id)
            avail_actions.append(avail_action)
        avail_u.append(np.array(avail_actions))
        # avail_u.append(avail_actions)
        avail_u_next = avail_u[1:]
        avail_u = avail_u[:-1]
        # Normalized reward
        # r = self.normalize_reward(r)

        # if step < self.episode_limit，padding
        # For clip by the episode limit
        for i in range(step, self.episode_limit):
            o.append(np.zeros((self.n_agents, self.obs_shape)))
            u.append(np.zeros([self.n_agents, 1])) # For num_steps we should pad it with -1
            s.append(np.zeros((self.n_agents, self.obs_shape)))
            r.append([[0., 0., 0.]])
            o_next.append(np.zeros((self.n_agents, self.obs_shape)))
            s_next.append(np.zeros((self.n_agents, self.state_shape)))
            u_onehot.append(np.zeros((self.n_agents, self.n_actions)))
            avail_u.append(np.zeros((self.n_agents, self.n_actions)))
            avail_u_next.append(np.zeros((self.n_agents, self.n_actions)))
            padded.append([1.])
            lead_ids.append([0])  # 任意
            terminate.append([True])

        episode = dict(o=o.copy(),
                       s=s.copy(),
                       u=u.copy(),
                       r=r.copy(),
                       avail_u=avail_u.copy(),
                       o_next=o_next.copy(),
                       s_next=s_next.copy(),
                       avail_u_next=avail_u_next.copy(),
                       u_onehot=u_onehot.copy(),
                       padded=padded.copy(),
                       terminated=terminate.copy(),
                       lead_ids=lead_ids.copy()
                       )
        # add episode dim
        for key in episode.keys():
            episode[key] = np.array([episode[key]])
        # episode['r'] = self.cal_return(episode['r']) # Only compute non-padding rewards to obtain the return
        if not evaluate:
            self.epsilon = epsilon
        if self.args.alg == 'maven':
            episode['z'] = np.array([maven_z.copy()])
        if evaluate and episode_num == self.args.evaluate_epoch - 1 and self.args.replay_dir != '':
            self.env.save_replay()
            self.env.close()
        end_step = step
        for agent in range(len(reward)):  # mean reward per episode
            episode_reward[agent] = episode_reward[agent]/end_step

        return episode, episode_reward, win_tag, end_step

    def cal_return(self, r):
        '''

        :param r: ndarray: (1, episode_limits, 1, n_agent)
        :return: reward to go, returns: ndarray: (1, episode_limits, 1, n_agent)
        '''
        num_steps = np.max(np.nonzero(r[0][:, 0, 0]))
        r = torch.from_numpy(r)
        returns = torch.FloatTensor(r.shape)  # (1, episode_limits, 1, n_agent) float type
        prev_return = 0

        for i in reversed(range(num_steps+1)):
            # print(returns.dtype)
            returns[:, i, :, :] = r[:, i, :, :] + self.args.gamma * prev_return
            prev_return = returns[:, i, :, :]

        returns = returns.numpy()
        return returns

    def normalize_reward(self, reward):
        '''

        :param reward: reward list of num_steps
        :return:
        '''
        r = np.array(reward)[:,0,0] # num_steps array for both agents
        r = (r-np.mean(r))/(np.std(r) + 1e-7)

        rewards = []
        for i in range(r.size):
            rewards.append([[r[i], r[i], r[i], r[i], r[i]]])
        return rewards

# RolloutWorker for communication
class CommRolloutWorker:
    def __init__(self, env, agents, args):
        self.env = env
        self.agents = agents
        self.episode_limit = args.episode_limit
        self.n_actions = args.n_actions
        self.n_agents = args.n_agents
        self.state_shape = args.state_shape
        self.obs_shape = args.obs_shape
        self.args = args

        self.epsilon = args.epsilon
        self.anneal_epsilon = args.anneal_epsilon
        self.min_epsilon = args.min_epsilon
        print('Init CommRolloutWorker')

    def generate_episode(self, episode_num=None, evaluate=False):
        if self.args.replay_dir != '' and evaluate and episode_num == 0:  # prepare for save replay
            self.env.close()
        o, u, r, s, avail_u, u_onehot, terminate, padded = [], [], [], [], [], [], [], []
        self.env.reset()
        terminated = False
        win_tag = False
        step = 0
        episode_reward = 0
        last_action = np.zeros((self.args.n_agents, self.args.n_actions))
        self.agents.policy.init_hidden(1)
        epsilon = 0 if evaluate else self.epsilon
        if self.args.epsilon_anneal_scale == 'episode':
            epsilon = epsilon - self.anneal_epsilon if epsilon > self.min_epsilon else epsilon
        if self.args.epsilon_anneal_scale == 'epoch':
            if episode_num == 0:
                epsilon = epsilon - self.anneal_epsilon if epsilon > self.min_epsilon else epsilon
        while not terminated and step < self.episode_limit:
            # time.sleep(0.2)
            obs = self.env.get_obs()
            state = self.env.get_state()
            actions, avail_actions, actions_onehot = [], [], []

            # get the weights of all actions for all agents
            weights = self.agents.get_action_weights(np.array(obs), last_action)

            # choose action for each agent
            for agent_id in range(self.n_agents):
                avail_action = self.env.get_avail_agent_actions(agent_id)
                action = self.agents.choose_action(weights[agent_id], avail_action, epsilon, evaluate)

                # generate onehot vector of th action
                action_onehot = np.zeros(self.args.n_actions)
                action_onehot[action] = 1
                actions.append(action)
                actions_onehot.append(action_onehot)
                avail_actions.append(avail_action)
                last_action[agent_id] = action_onehot

            reward, terminated, info = self.env.step(actions)
            win_tag = True if terminated and 'battle_won' in info and info['battle_won'] else False
            o.append(obs)
            s.append(state)
            u.append(np.reshape(actions, [self.n_agents, 1]))
            u_onehot.append(actions_onehot)
            avail_u.append(np.array(avail_actions))
            # avail_u.append(avail_actions)
            r.append([reward])
            terminate.append([terminated])
            padded.append([0.])
            episode_reward += reward
            step += 1
            # if terminated:
            #     time.sleep(1)
            if self.args.epsilon_anneal_scale == 'step':
                epsilon = epsilon - self.anneal_epsilon if epsilon > self.min_epsilon else epsilon
        # last obs
        o.append(obs)
        s.append(state)
        o_next = o[1:]
        s_next = s[1:]
        o = o[:-1]
        s = s[:-1]
        # get avail_action for last obs，because target_q needs avail_action in training
        avail_actions = []
        for agent_id in range(self.n_agents):
            avail_action = self.env.get_avail_agent_actions(agent_id)
            avail_actions.append(avail_action)
        avail_u.append(avail_actions)
        avail_u_next = avail_u[1:]
        avail_u = avail_u[:-1]

        # if step < self.episode_limit，padding
        for i in range(step, self.episode_limit):
            o.append(np.zeros((self.n_agents, self.obs_shape)))
            u.append(np.zeros([self.n_agents, 1]))
            s.append(np.zeros(self.state_shape))
            r.append([0.])
            o_next.append(np.zeros((self.n_agents, self.obs_shape)))
            s_next.append(np.zeros(self.state_shape))
            u_onehot.append(np.zeros((self.n_agents, self.n_actions)))
            avail_u.append(np.zeros((self.n_agents, self.n_actions)))
            avail_u_next.append(np.zeros((self.n_agents, self.n_actions)))
            padded.append([1.])
            terminate.append([1.])

        episode = dict(o=o.copy(),
                       s=s.copy(),
                       u=u.copy(),
                       r=r.copy(),
                       avail_u=avail_u.copy(),
                       o_next=o_next.copy(),
                       s_next=s_next.copy(),
                       avail_u_next=avail_u_next.copy(),
                       u_onehot=u_onehot.copy(),
                       padded=padded.copy(),
                       terminated=terminate.copy()
                       )
        # add episode dim
        for key in episode.keys():
            episode[key] = np.array([episode[key]])
        if not evaluate:
            self.epsilon = epsilon
            # print('Epsilon is ', self.epsilon)
        if evaluate and episode_num == self.args.evaluate_epoch - 1 and self.args.replay_dir != '':
            self.env.save_replay()
            self.env.close()
        return episode, episode_reward, win_tag
