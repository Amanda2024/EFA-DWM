from runner import Runner
# from games_new.seek_modify_env import seek_env
import os, sys
import torch
import random
sys.path.insert(1, os.path.join(sys.path[0], 'multiagent-particle-envs'))
import argparse

from multiagent.environment import MultiAgentEnv
from multiagent.policy import RandomPolicy
import multiagent.scenarios as scenarios

from common.arguments import get_common_args, get_coma_args, get_mixer_args, get_centralv_args, get_reinforce_args, \
    get_commnet_args, get_g2anet_args
from functools import reduce
from tensorboardX import SummaryWriter

from datetime import datetime, timedelta
import time
import json
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    iter_end_step = []
    #for i in range(10):
    for i in range(0, 10, 1):
        torch.manual_seed(i)
        torch.cuda.manual_seed(i)
        np.random.seed(seed=i)
        random.seed(i)
        args = get_common_args()   # Namespace(action_dim=5, alg='ssg_vdn_two', anneal_epsilon=3.0000000000000005e-06, batch_size=32, buffer_size=5000, cuda=False, difficulty='7', entropy_coefficient=0.001, epsilon=0.2, epsilon_anneal_scale='step', evaluate_cycle=100, evaluate_epoch=20, game_version='latest', gamma=0.99, grad_norm_clip=10, hyper_hidden_dim=64, lambda_mi=0.001, lambda_nopt=1, lambda_opt=1, lambda_ql=1, last_action=True, learn=True, load_model=False, log_dir='logs/', lr=0.0005, map='3m', min_epsilon=0.05, model_dir='./model', n_episodes=1, n_epoch=20000, noise_dim=16, optimizer='RMS', qmix_hidden_dim=32, qtran_hidden_dim=64, replay_dir='', result_dir='./result', reuse_network=True, rnn_hidden_dim=64, save_cycle=5000, scenario='simple_spread.py', seed=123, step_mul=8, target_update_cycle=200, train_steps=1, two_hyper_layers=False)
        if args.alg.find('coma') > -1:
            args = get_coma_args(args)
        elif args.alg.find('central_v') > -1:
            args = get_centralv_args(args)
        elif args.alg.find('reinforce') > -1:
            args = get_reinforce_args(args)
        else:
            args = get_mixer_args(args)
        if args.alg.find('commnet') > -1:
            args = get_commnet_args(args)
        if args.alg.find('g2anet') > -1:
            args = get_g2anet_args(args)
        # env = StarCraft2Env(map_name=args.map,
        #                     step_mul=args.step_mul,
        #                     difficulty=args.difficulty,
        #                     game_version=args.game_version,
        #                     replay_dir=args.replay_dir)
        # env = seek_env()
        scenario = scenarios.load(args.scenario).Scenario()
        # create world
        world = scenario.make_world()
        # create multiagent environment
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, info_callback=None,
                            shared_viewer=False)  # 2
        # render call to create viewer window (necessary only for interactive policies)
        # env.render()

        policies = [RandomPolicy(env, i) for i in range(env.n)]
        obs_n = env.reset()

        # env_info = env.get_env_info()
        args.n_actions = env.action_space[0].n  # The same shape for each agent # 5
        # args.n_actions = env.game.joint_action_space[0][0]  # The same shape for each agent
        args.n_agents = env.n
        obs_shape = [len(list(obs_n[0]))]  # The only agent in the simple.py env default
        # obs_shape = list(env.game.get_grid_many_obs_space(range(env.game.n_player))[0])
        # args.state_shape = 4 * reduce(lambda x, y: x * y, obs_shape)
        args.state_shape = reduce(lambda x, y: x * y, obs_shape)  # For modified env (3*3*5)  #  把list内int相乘降维为int
        args.obs_shape = args.state_shape
        args.episode_limit = 25
        runner = Runner(env, args)

        # For more simple logs
        cur_time = datetime.now() + timedelta(hours=0)
        args.log_dir = "logs/asg_vdn_leader_maintain/elect_update/" + cur_time.strftime("[%m-%d]%H.%M.%S")

        writer = SummaryWriter(logdir=args.log_dir)
        if args.learn:
            end_step = runner.run(i, writer)
            iter_end_step.append(end_step)
        else:
            win_rate, _ = runner.evaluate()
            print('The win rate of {} is  {}'.format(args.alg, win_rate))
            break
        # runner.replay()
        env.close()

    ### 存为文件
   # now = time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime(time.time()))
   # filename = "logs/" + "end_step_" + now
   # with open(filename, 'a+') as file_obj:
   #     json.dump(iter_end_step, file_obj)
