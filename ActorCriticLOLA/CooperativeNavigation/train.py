import gym
from IAC_LOLA import ActorCritic
import torch

from tensorboardX import SummaryWriter
from datetime import datetime, timedelta
import numpy as np
import sys
import os
sys.path.insert(1, os.path.join(sys.path[0], 'multiagent_particle_envs'))
from multiagent.environment import MultiAgentEnv
import multiagent.scenarios as scenarios
import random

def main(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed=seed)
    random.seed(seed)
    # env = gym.make("CartPole-v0")
    scenario = 'simple_spread.py'
    scenario = scenarios.load(scenario).Scenario()
    # create world
    world = scenario.make_world()
    # create multiagent environment
    env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, info_callback=None,
                        shared_viewer=False)  # 2
    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = 'cpu'
    agent1 = ActorCritic(args=(12, 5, device, 1e-3, 1e-3, 25)).to(device)
    agent2 = ActorCritic(args=(12, 5, device, 1e-3, 1e-3, 25)).to(device)
    agent1_ = ActorCritic(args=(12, 5, device, 1e-3, 1e-3, 25)).to(device)
    agent2_ = ActorCritic(args=(12, 5, device, 1e-3, 1e-3, 25)).to(device)

    # For more simple logs
    cur_time = datetime.now() + timedelta(hours=0)
    log_dir = "logs/" + cur_time.strftime("[%m-%d]%H.%M.%S") + "_seed_" + str(seed)
    writer = SummaryWriter(logdir=log_dir)

    for i in range(30000):
        # agent1_ = agent1.copy().detach().requires_grad_(True)
        # agent2_ = agent2.copy().detach().requires_grad_(True)
        agent1_.load_state_dict(agent1.state_dict())
        agent2_.load_state_dict(agent2.state_dict())

        n_lookaheads = 1
        for k in range(n_lookaheads):
            grad2 = agent1.in_lookahead(env, agent2)
            grad1 = agent2.in_lookahead(env, agent1)

            j = 0
            for p in agent2_.actor.parameters():
                p.data = p.data - 0.3 * grad2[j]
                j = j + 1

            j = 0
            for p in agent1_.actor.parameters():
                p.data = p.data - 0.3 * grad1[j]
                j = j + 1

        a_loss_1, v_loss_1 = agent1.out_lookahead(env, agent2_)
        a_loss_2, v_loss_2 = agent2.out_lookahead(env, agent1_)

        # print
        if (i + 1) % 20 == 0:
            scores = []
            for _ in range(10):
                score = 0.
                s = env.reset()
                for t in range(25):
                    s = torch.FloatTensor(s).to(device)
                    action1, log_p1 = agent1.select_action(s[0])
                    action1_one_hot = agent1.to_one_hot(action1)
                    action2, log_p2 = agent2.select_action(s[1])
                    action2_one_hot = agent2.to_one_hot(action2)
                    p_actions_onehot = [action1_one_hot, action2_one_hot]
                    s_next, reward, done, info = env.step(p_actions_onehot)
                    reward = torch.tensor(reward[0])
                    score += reward
                    s = s_next
                    if done:
                        break
                scores.append(score)

            writer.add_scalar('actor 1 loss', a_loss_1, i)
            writer.add_scalar('actor 2 loss', a_loss_2, i)
            writer.add_scalar('critic 1 loss', v_loss_1, i)
            writer.add_scalar('critic 2 loss', v_loss_2, i)
            writer.add_scalar('episode_rewards', np.mean(scores), i)

            print("Seed :", seed, "Epoch :", i, "episode_rewards :", np.mean(scores))




if __name__ == "__main__":
    for i in range(10):
        main(i)
