import gfootball.env as football_env
import time, pprint, importlib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import torch.multiprocessing as mp
from tensorboardX import SummaryWriter
import random

def write_summary(writer, arg_dict, summary_queue, n_game, loss_lst, optimization_step, self_play_board, win_evaluation, score_evaluation):
    win, score, tot_reward, game_len = [], [], [], []
    loop_t, forward_t, wait_t = [], [], []

    for i in range(arg_dict["summary_game_window"]):
        game_data = summary_queue.get()
        a, b, c, d, opp_num, t1, t2, t3 = game_data
        if arg_dict["env"] == "11_vs_11_kaggle":  ## 这是干啥的
            if opp_num in self_play_board:
                self_play_board[opp_num].append(a)
            else:
                self_play_board[opp_num] = [a]

        if 'env_evaluation' in arg_dict and opp_num == arg_dict['env_evaluation']:
            win_evaluation.append(a)
            score_evaluation.append(b)
        else:
            win.append(a)
            score.append(b)
            tot_reward.append(c)
            game_len.append(d)
            loop_t.append(t1)
            forward_t.append(t2)
            wait_t.append(t3)

    writer.add_scalar('game/win_rate', float(np.mean(win)), n_game)
    writer.add_scalar('game/score', float(np.mean(score)), n_game)
    writer.add_scalar('game/reward', float(np.mean(tot_reward)), n_game)
    writer.add_scalar('game/game_len', float(np.mean(game_len)), n_game)
    writer.add_scalar('game/loss', np.mean(loss_lst), n_game)
    writer.add_scalar('train/step', float(optimization_step), n_game)
    writer.add_scalar('time/loop', float(np.mean(loop_t)), n_game)
    writer.add_scalar('time/forward', float(np.mean(forward_t)), n_game)
    writer.add_scalar('time/wait', float(np.mean(wait_t)), n_game)
    writer.add_scalar('train/loss', np.mean(loss_lst), n_game)
    # writer.add_scalar('train/pi_loss', np.mean(pi_loss_lst), n_game)
    # writer.add_scalar('train/v_loss', np.mean(v_loss_lst), n_game)
    # writer.add_scalar('train/entropy', np.mean(entropy_lst), n_game)
    # writer.add_scalar('train/move_entropy', np.mean(move_entropy_lst), n_game)

    mini_window = max(1, int(arg_dict['summary_game_window'] / 3))
    if len(win_evaluation) >= mini_window:
        writer.add_scalar('game/win_rate_evaluation', float(np.mean(win_evaluation)), n_game)
        writer.add_scalar('game/score_evaluation', float(np.mean(score_evaluation)), n_game)
        win_evaluation, score_evaluation = [], []

    for opp_num in self_play_board:
        if len(self_play_board[opp_num]) >= mini_window:
            label = 'self_play/' + opp_num
            writer.add_scalar(label, np.mean(self_play_board[opp_num][:mini_window]), n_game)
            self_play_board[opp_num] = self_play_board[opp_num][mini_window:]

    return win_evaluation, score_evaluation


def write_summary_simplified(writer, arg_dict, summary_queue, n_game, loss_lst, \
                             optimization_step):
    win, score, tot_reward, game_len = [], [], [], []
    loop_t, forward_t, wait_t = [], [], []

    writer.add_scalar('game/win_rate', float(np.mean(win)), n_game)
    writer.add_scalar('game/score', float(np.mean(score)), n_game)
    writer.add_scalar('game/reward', float(np.mean(tot_reward)), n_game)
    writer.add_scalar('game/game_len', float(np.mean(game_len)), n_game)
    writer.add_scalar('train/step', float(optimization_step), n_game)
    writer.add_scalar('time/loop', float(np.mean(loop_t)), n_game)
    writer.add_scalar('time/forward', float(np.mean(forward_t)), n_game)
    writer.add_scalar('time/wait', float(np.mean(wait_t)), n_game)
    writer.add_scalar('train/loss', np.mean(loss_lst), n_game)

    mini_window = max(1, int(arg_dict['summary_game_window'] / 3))

def Random_Queue(queue, arg_dict):
    queue_list = []
    queue_size = queue.qsize()
    ranint = random.sample(range(queue_size), arg_dict["buffer_size"] * arg_dict["batch_size"])
    for i in range(queue_size):
        rollout = queue.get()
        queue_list.append(rollout)
    random_list = [queue_list[index] for index in ranint]

    for i in list(set(range(queue_size)).difference(ranint)):
        queue.put(queue_list[i])
    return random_list

def get_data(queue, arg_dict, model):
    # print("--------------------:", queue.qsize())
    data = []
    # random_list = Random_Queue(queue, arg_dict)
    for i in range(arg_dict["buffer_size"]):  # 6
        mini_batch_np = []
        for j in range(arg_dict["batch_size"]): # 3
            # rollout = random_list[i*arg_dict["buffer_size"] + j]
            rollout = queue.get()
            mini_batch_np.append(rollout)
        mini_batch = model.make_batch(mini_batch_np)  # mini_batch_np( 32, 60*[transition, transition]) --> mini_batch含8个tuple，除s外每个的shape(120,32,1)  # 8 指的是：s, a, m, r, s_prime, done_mask, prob, need_move
        data.append(mini_batch)
    return data


def learner(center_model_1, center_model_2, center_mixer, queue, data_all_queue, signal_queue, summary_queue, arg_dict):
    print("Learner process started")
    imported_model = importlib.import_module("models.agents." + arg_dict["model"])
    imported_mixer = importlib.import_module("models.mixers." + arg_dict["mixer_net"])
    imported_algo = importlib.import_module("algos." + arg_dict["algorithm"])
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = "cpu"
    # 根据参数决定RNN的输入维度
    input_shape = arg_dict["state_shape"]
    if arg_dict["last_action"]:
        input_shape += arg_dict["n_actions"]
    if arg_dict["reuse_network"]:
        input_shape += arg_dict["n_agents"]

    model1 = imported_model.RNNAgent(input_shape, arg_dict, device)
    model2 = imported_model.RNNAgent(input_shape, arg_dict, device)
    model1.load_state_dict(center_model_1.state_dict())
    model1.optimizer.load_state_dict(center_model_1.optimizer.state_dict())
    model2.load_state_dict(center_model_2.state_dict())
    model2.optimizer.load_state_dict(center_model_2.optimizer.state_dict())

    mixer = imported_mixer.VDNMixer()
    mixer.load_state_dict(center_mixer.state_dict())

    algo1 = imported_algo.QLearner(arg_dict, model1, model2, mixer)
    algo2 = imported_algo.QLearner(arg_dict, model2, model1, mixer)


    # if torch.cuda.is_available():
    #     algo.cuda()

    writer = SummaryWriter(logdir=arg_dict["log_dir"])
    optimization_step = 0
    if "optimization_step" in arg_dict:
        optimization_step = arg_dict["optimization_step"]
    last_saved_step = optimization_step
    n_game = 0
    loss_lst, pi_loss_lst, v_loss_lst, entropy_lst, move_entropy_lst = [], [], [], [], []
    self_play_board = {}
    win_evaluation, score_evaluation = [], []

    while True:
        if queue.qsize() > arg_dict["batch_size"] * arg_dict["buffer_size"] * arg_dict["rich_data_scale"]:
            if (optimization_step % arg_dict["model_save_interval"] == 0):  # rjq  save model
                path = arg_dict["log_dir"] + "/model_" + str(optimization_step) + ".tar"
                algo1.save_models(path, model1, mixer)
                algo2.save_models(path, model2, mixer)

            signal_queue.put(1)
            data = get_data(queue, arg_dict, model1)
            print("data loaded……")
            # before_model.load_state_dict(model.state_dict())
            loss_1 = algo1.train(model1, mixer, data)
            loss_2 = algo2.train(model2, mixer, data)

            # loss, pi_loss, v_loss, entropy, move_entropy = algo.train(model, data)
            # optimization_step += arg_dict["batch_size"] * arg_dict["buffer_size"] * arg_dict["k_epoch"]
            optimization_step += 1
            print("step :", optimization_step, "loss", loss_1, "data_q", queue.qsize(), "summary_q",
                  summary_queue.qsize())

            loss_1 = loss_1.cpu().detach().numpy()
            loss_lst.append(loss_1)
            center_model_1.load_state_dict(model1.state_dict())  # How to load the q-agent and mixer together
            center_model_2.load_state_dict(model2.state_dict())  # How to load the q-agent and mixer together
            center_mixer.load_state_dict(mixer.state_dict())

            # if queue.qsize() > arg_dict["batch_size"] * arg_dict["buffer_size"]:
            #     print("warning. data remaining. queue size : ", queue.qsize())

            if summary_queue.qsize() > arg_dict["summary_game_window"]:
                # write_summary_simplified(writer, arg_dict, summary_queue, n_game, loss_lst, optimization_step)
                win_evaluation, score_evaluation = write_summary(writer, arg_dict, summary_queue, n_game, loss_lst,
                                                                 optimization_step, self_play_board, win_evaluation, score_evaluation)

                loss_lst, pi_loss_lst, v_loss_lst, entropy_lst, move_entropy_lst = [], [], [], [], []

                data_all_queue.put(n_game)
                n_game += arg_dict["summary_game_window"]

            _ = signal_queue.get()

        else:
            time.sleep(0.1)
