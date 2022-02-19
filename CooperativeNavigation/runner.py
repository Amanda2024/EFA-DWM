import numpy as np
import os
from common.rollout import RolloutWorker, CommRolloutWorker
from agent.agent import Agents, CommAgents
from common.replay_buffer import ReplayBuffer
import matplotlib.pyplot as plt

from tensorboardX import SummaryWriter

class Runner:
    def __init__(self, env, args):
        self.env = env

        if args.alg.find('commnet') > -1 or args.alg.find('g2anet') > -1:  # communication agent
            self.agents = CommAgents(args)
            self.rolloutWorker = CommRolloutWorker(env, self.agents, args)
        else:  # no communication agent
            self.agents = Agents(args)
            self.rolloutWorker = RolloutWorker(env, self.agents, args)
        if args.learn and args.alg.find('coma') == -1 and args.alg.find('central_v') == -1 and args.alg.find(
                'reinforce') == -1:  # these 3 algorithms are on-poliy
            self.buffer = ReplayBuffer(args)
        self.args = args
        self.win_rates = []
        self.episode_rewards = []

        # 用来保存plt和pkl
        self.save_path = self.args.result_dir + '/' + args.alg + '/' + args.map
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

    def run(self, num, writer):
        train_steps = 0
        # print('Run {} start'.format(num))

        # writer = SummaryWriter(logdir=self.args.log_dir)

        end_steps_epoch = []
        step_indx = 0
        for epoch in range(self.args.n_epoch):
            print('Run {}, train epoch {}'.format(num, epoch))
            if epoch % self.args.evaluate_cycle == 0:
                win_rate, episode_reward = self.evaluate()
                # print('win_rate is ', win_rate)
                self.win_rates.append(win_rate)
                self.episode_rewards.append(episode_reward)
                # if self.args.scenario == 'simple_spread.py':
                #     self.plt_svg(num)
                # else:
                #     self.plt(num)

            episodes = []
            end_steps = []
            episode_rewards = [0] * self.args.n_agents
            # 收集self.args.n_episodes个episodes
            for episode_idx in range(self.args.n_episodes):
                episode, episode_reward, _, end_step = self.rolloutWorker.generate_episode(episode_idx)
                end_steps.append(end_step)
                episodes.append(episode)

                for agent in range(len(episode_reward)):
                    episode_rewards[agent] += episode_reward[agent]

                # print(_)
            # episode的每一项都是一个(1, episode_len, n_agents, 具体维度)四维数组，下面要把所有episode的的obs拼在一起
            episode_batch = episodes[0]
            episodes.pop(0)
            for episode in episodes:
                for key in episode_batch.keys():
                    episode_batch[key] = np.concatenate((episode_batch[key], episode[key]), axis=0)
            if self.args.alg.find('coma') > -1 or self.args.alg.find('central_v') > -1 or self.args.alg.find(
                    'reinforce') > -1:
                self.agents.train(episode_batch, train_steps, self.rolloutWorker.epsilon)
                train_steps += 1
            else:
                self.buffer.store_episode(episode_batch)
                loss = []
                for train_step in range(self.args.train_steps):
                    mini_batch = self.buffer.sample(min(self.buffer.current_size, self.args.batch_size))
                    updated_loss = self.agents.train(mini_batch, train_steps)
                    updated_loss = updated_loss.detach().numpy()
                    loss.append(updated_loss)
                    train_steps += 1
                writer.add_scalar('loss', np.mean(loss), train_steps)

            end_steps_epoch += end_steps
            step_indx += 1
            writer.add_scalar('mean_end_step', np.mean(end_steps), step_indx)
            writer.add_scalar('episode_rewards', np.mean(episode_rewards), step_indx)  ## rjq0109 添加评估的reward  ### episode_rewards: <class 'list'>: [-6.37927245923192, -6.37927245923192]
            writer.add_scalar('episode_rewards_mean', np.mean(self.episode_rewards), step_indx)  # later add 0605

       # if self.args.scenario == 'simple_spread.py':
       #     self.plt_svg(num)
       # else:
       #     self.plt(num)

        return end_steps_epoch


    def replay(self):
        buffer_state_set = self.buffer.buffers['o']
        buffer_action_set = self.buffer.buffers['u']
        # print("Buffer size", len(buffer_action_set))
        self.env.game.replay(self.args.n_epoch, buffer_state_set, buffer_action_set, self.args.episode_limit)

    def evaluate(self):
        win_number = 0
        # episode_rewards = 0
        episode_rewards = [0] * self.args.n_agents  # list for two agents
        for epoch in range(self.args.evaluate_epoch):
            _, episode_reward, win_tag, _ = self.rolloutWorker.generate_episode(epoch, evaluate=True)
            for agent in range(len(episode_reward)):
                episode_rewards[agent] += episode_reward[agent]
            if win_tag:
                win_number += 1
        eps_rewards = [x / self.args.evaluate_epoch for x in episode_rewards]
        # return win_number / self.args.evaluate_epoch, episode_rewards / self.args.evaluate_epoch
        return win_number / self.args.evaluate_epoch, eps_rewards

    def plt(self, num):
        plt.figure()
        plt.axis([0, self.args.n_epoch, 0, 100])
        plt.cla()
        plt.subplot(2, 1, 1)
        plt.plot(range(len(self.win_rates)), self.win_rates)
        plt.xlabel('epoch*{}'.format(self.args.evaluate_cycle))
        plt.ylabel('win_rate')

        plt.subplot(2, 1, 2)
        plt.plot(range(len(self.episode_rewards)), self.episode_rewards)
        plt.xlabel('epoch*{}'.format(self.args.evaluate_cycle))
        plt.ylabel('episode_rewards')
        if self.args.scenario == 'simple_spread.py':
            plt.title('Cooperation Navigation ($n$=2)')

        plt.savefig(self.save_path + '/plt_{}.png'.format(num), format='png')
        np.save(self.save_path + '/win_rates_{}'.format(num), self.win_rates)
        np.save(self.save_path + '/episode_rewards_{}'.format(num), self.episode_rewards)

    def plt_svg(self, num):
        plt.figure()
        plt.axis([0, self.args.n_epoch, 0, 100])
        plt.cla()
        # plt.subplot(2, 1, 1)
        # plt.plot(range(len(self.win_rates)), self.win_rates)
        # plt.xlabel('epoch*{}'.format(self.args.evaluate_cycle))
        # plt.ylabel('win_rate')

        # plt.subplot(2, 1, 2)
        plt.plot(range(len(self.episode_rewards)), self.episode_rewards)
        plt.xlabel('epoch*{}'.format(self.args.evaluate_cycle))
        plt.ylabel('episode_rewards')
        if self.args.scenario == 'simple_spread.py':
            plt.title('Cooperation Navigation ($n$=2)')

        plt.savefig(self.save_path + '/plt_{}.svg'.format(num), format='svg')
        np.save(self.save_path + '/win_rates_{}'.format(num), self.win_rates)
        np.save(self.save_path + '/episode_rewards_{}'.format(num), self.episode_rewards)
