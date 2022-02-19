import torch
import os
from network.base_net import RNN
from network.follower_net import Follower_RNN
from network.election_net import Election_Model
from network.qmix_net import QMixNet
from network.vdn_net import VDNNet
import numpy as np
import torch.nn.functional as F
from common.utils import gumbel_softmax
# from torchsummary import summary

from tensorboardX import SummaryWriter

class ASG_VDN:
    def __init__(self, args):
        self.n_actions = args.n_actions
        self.n_agents = args.n_agents
        self.state_shape = args.state_shape
        self.obs_shape = args.obs_shape
        input_shape = self.obs_shape
        # 根据参数决定RNN的输入维度
        if args.last_action:
            input_shape += self.n_actions
        if args.reuse_network:
            input_shape += self.n_agents

        # 神经网络
        self.election = Election_Model(self.obs_shape, args)
        self.target_election = Election_Model(self.obs_shape, args)
        self.eval_rnn = RNN(input_shape, args)  # 每个agent选动作的网络
        self.target_rnn = RNN(input_shape, args)
        self.eval_follower = Follower_RNN(input_shape, args)  # Follower的RNN
        self.target_follower = Follower_RNN(input_shape, args)  # For target mix net
        self.eval_vdn_net = VDNNet()  # 把agentsQ值加起来的网络
        self.target_vdn_net = VDNNet()
        self.args = args
        if self.args.cuda:
            self.election.cuda()
            self.target_election.cuda()
            self.eval_rnn.cuda()
            self.target_rnn.cuda()
            self.eval_follower.cuda()
            self.target_follower.cuda()
            self.eval_vdn_net.cuda()
            self.target_vdn_net.cuda()
        self.model_dir = args.model_dir + '/' + args.alg + '/' + args.map
        # 如果存在模型则加载模型
        if self.args.load_model:
            if os.path.exists(self.model_dir + '/rnn_net_params.pkl'):
                path_election = self.model_dir + '/election_model_params.pkl'
                path_rnn = self.model_dir + '/rnn_net_params.pkl'
                path_follower = self.model_dir + '/follower_net_params.pkl'
                path_vdn = self.model_dir + '/vdn_net_params.pkl'
                map_location = 'cuda:0' if self.args.cuda else 'cpu'
                self.election.load_state_dict(torch.load(path_election, map_location=map_location))
                self.eval_rnn.load_state_dict(torch.load(path_rnn, map_location=map_location))
                self.eval_follower.load_state_dict(torch.load(path_follower, map_location=map_location))
                self.eval_vdn_net.load_state_dict(torch.load(path_vdn, map_location=map_location))
                print('Successfully load the model: {} and {}'.format(path_rnn, path_vdn))
            else:
                raise Exception("No model!")

        # 让target_net和eval_net的网络参数相同
        self.target_rnn.load_state_dict(self.eval_rnn.state_dict())
        self.target_follower.load_state_dict(self.eval_follower.state_dict())
        self.target_vdn_net.load_state_dict(self.eval_vdn_net.state_dict())

        self.eval_parameters = list(self.eval_vdn_net.parameters()) + list(self.eval_rnn.parameters()) + \
                               list(self.eval_follower.parameters()) + list(self.election.parameters())
        if args.optimizer == "RMS":
            self.optimizer = torch.optim.RMSprop(self.eval_parameters, lr=args.lr)

        # 执行过程中，要为每个agent都维护一个eval_hidden
        # 学习过程中，要为每个episode的每个agent都维护一个eval_hidden、target_hidden
        self.election_hidden = None
        self.target_election_hidden = None
        self.eval_hidden = None
        self.target_hidden = None
        self.eval_follower_hidden = None
        self.target_follower_hidden = None
        print('Init alg ASG_VDN')

    def learn(self, batch, max_episode_len, train_step, epsilon=None):  # train_step表示是第几次学习，用来控制更新target_net网络的参数
        '''
        在learn的时候，抽取到的数据是四维的，四个维度分别为 1——第几个episode 2——episode中第几个transition
        3——第几个agent的数据 4——具体obs维度。因为在选动作时不仅需要输入当前的inputs，还要给神经网络输入hidden_state，
        hidden_state和之前的经验相关，因此就不能随机抽取经验进行学习。所以这里一次抽取多个episode，然后一次给神经网络
        传入每个episode的同一个位置的transition
        '''
        episode_num = batch['o'].shape[0]
        self.init_hidden(episode_num)
        for key in batch.keys():  # 把batch里的数据转化成tensor
            if key == 'u':
                batch[key] = torch.tensor(batch[key], dtype=torch.long)
            else:
                batch[key] = torch.tensor(batch[key], dtype=torch.float32)
        s, s_next, u, r, avail_u, avail_u_next, terminated = batch['s'], batch['s_next'], batch['u'], \
                                                             batch['r'], batch['avail_u'], batch['avail_u_next'], \
                                                             batch['terminated']
        mask = 1 - batch["padded"].float()  # 用来把那些填充的经验的TD-error置0，从而不让它们影响到学习

        # 得到每个agent对应的Q值，维度为(episode个数, max_episode_len， n_agents， n_actions)
        q_evals, q_targets = self.get_q_values(batch, max_episode_len)
        if self.args.cuda:
            s = s.cuda()
            u = u.cuda()
            r = r.cuda()
            s_next = s_next.cuda()
            terminated = terminated.cuda()
            mask = mask.cuda()

        q_values_bk = q_evals
        # 取每个agent动作对应的Q值，并且把最后不需要的一维去掉，因为最后一维只有一个值了
        q_evals = torch.gather(q_evals, dim=3, index=u).squeeze(3)

        # 得到target_q
        q_targets[avail_u_next == 0.0] = - 9999999
        q_targets = q_targets.max(dim=3)[0]

        q_total_eval = self.eval_vdn_net(q_evals)  # (1, 50, 1)
        # q_total_eval = self.eval_qmix_net(q_evals, s)  # (1, 50, 1)
        q_total_target = self.target_vdn_net(q_targets)  # (1, 50, 1)

        r = 0.5 * torch.sum(r, dim=-1).reshape(q_total_eval.shape[0], -1, 1)  # Total reward for each agent
        targets = r + self.args.gamma * q_total_target * (1 - terminated)

        td_error = (q_total_eval - targets.detach())
        masked_td_error = mask * td_error  # 抹掉填充的经验的td_error

        ########  加入加权vdn TODO:rjq add weight
        w_to_use = self.args.w
        if not self.args.w_vdn:
            loss = torch.mean(masked_td_error ** 2)
            # loss = (masked_td_error ** 2).sum() / mask.sum()
        else:
            cur_max_actions = q_values_bk.max(dim=3)[0]
            is_max_action = (u.squeeze(-1) == cur_max_actions).min(dim=2)[0]
            qtot_larger = targets > q_total_eval  # (1,25,1)
            ws = torch.ones_like(td_error) * w_to_use  # (1,25,1)
            ws = torch.where(is_max_action.unsqueeze(-1) | qtot_larger, torch.ones_like(td_error) * 1,
                             ws)  # Target is greater than current max  (condition, x, y)
            w_to_use = ws.mean().item()
            loss = torch.mean(ws * masked_td_error ** 2)
            # loss = (ws * (masked_td_error ** 2)).sum() / mask.sum()

        # 不能直接用mean，因为还有许多经验是没用的，所以要求和再比真实的经验数，才是真正的均值
        # loss = torch.mean(masked_td_error ** 2)
        # loss = (masked_td_error ** 2).sum() / mask.sum()
        # loss = loss.requires_grad_()
        # state_dict1 = self.eval_rnn.state_dict()['fc1.weight']
        # state_dict2 = self.election.state_dict()['fc1.weight']
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.eval_parameters, self.args.grad_norm_clip)
        self.optimizer.step()

        # state_dict11 = self.eval_rnn.state_dict()['fc1.weight']
        # state_dict22 = self.election.state_dict()['fc1.weight']

        # writer = SummaryWriter(comment='test_your_comment', filename_suffix="_test_your_filename_suffix")
        #
        # # 模型
        # fake_img = torch.randn(1, 3, 32, 32)  # 生成假的图片作为输入
        #
        # writer.add_graph(self.eval_rnn,self.eval_hidden, fake_img)  # 模型及模型输入数据
        #
        # writer.close()

        if train_step > 0 and train_step % self.args.target_update_cycle == 0:
            self.target_rnn.load_state_dict(self.eval_rnn.state_dict())
            self.target_follower.load_state_dict(self.eval_follower.state_dict())
            self.target_vdn_net.load_state_dict(self.eval_vdn_net.state_dict())

        return loss  # log the loss after updating

    def _get_inputs(self, batch, transition_idx):
        # 取出所有episode上该transition_idx的经验，u_onehot要取出所有，因为要用到上一条
        obs, obs_next, u_onehot, lead_ids = batch['o'][:, transition_idx], \
                                            batch['o_next'][:, transition_idx], \
                                            batch['u_onehot'][:], batch['lead_ids'][:, transition_idx]  ##(1,3,18),#(1,3,18),#(1, 25, 3, 5), # (1,1)
        episode_num = obs.shape[0]
        inputs, inputs_next = [], []
        inputs.append(obs)
        inputs_next.append(obs_next)
        # 给obs添加上一个动作、agent编号

        if self.args.last_action:
            if transition_idx == 0:  # 如果是第一条经验，就让前一个动作为0向量
                inputs.append(torch.zeros_like(u_onehot[:, transition_idx]))
            else:
                inputs.append(u_onehot[:, transition_idx - 1])
            inputs_next.append(u_onehot[:, transition_idx])
        if self.args.reuse_network:
            # 因为当前的obs三维的数据，每一维分别代表(episode编号，agent编号，obs维度)，直接在dim_1上添加对应的向量
            # 即可，比如给agent_0后面加(1, 0, 0, 0, 0)，表示5个agent中的0号。而agent_0的数据正好在第0行，那么需要加的
            # agent编号恰好就是一个单位矩阵，即对角线为1，其余为0
            inputs.append(torch.eye(self.args.n_agents).unsqueeze(0).expand(episode_num, -1, -1))
            inputs_next.append(torch.eye(self.args.n_agents).unsqueeze(0).expand(episode_num, -1, -1))
        # 要把obs中的三个拼起来，并且要把episode_num个episode、self.args.n_agents个agent的数据拼成40条(40,96)的数据，
        # 因为这里所有agent共享一个神经网络，每条数据中带上了自己的编号，所以还是自己的数据
        # inputs = torch.cat([x.reshape(episode_num * self.args.n_agents, -1) for x in inputs], dim=1)
        inputs = torch.cat([x.reshape(episode_num, self.args.n_agents, -1) for x in inputs], dim=-1)
        # inputs_next = torch.cat([x.reshape(episode_num * self.args.n_agents, -1) for x in inputs_next], dim=1)
        inputs_next = torch.cat([x.reshape(episode_num, self.args.n_agents, -1) for x in inputs_next], dim=-1)
        return inputs, inputs_next, obs, obs_next, lead_ids

    def get_q_values(self, batch, max_episode_len):  # New for ssg with asymmetric structure
        episode_num = batch['o'].shape[0]
        q_evals, q_targets = [], []
        for transition_idx in range(max_episode_len):
            inputs, inputs_next, obs, _, leader_id_s = self._get_inputs(batch, transition_idx)  # 给obs加last_action、agent_id
            leader_id = []
            election_hidden = []

            # for i in range(obs.shape[0]):
            #     leader_id.append(leader_id_s[i][0].item())  # 选取前传的leader_id, 这个是不对的，这样子无法更新election_net
            for i in range(obs.shape[0]):
                leader_id_s, election_hidden_s = self.get_leader_id(obs[i], self.election_hidden[i])
                leader_id.append(leader_id_s)
                election_hidden.append(election_hidden_s)
            self.election_hidden = torch.vstack(election_hidden)



            follower_indices = [[i for i in range(self.args.n_agents)]] * episode_num
            for i in range(len(leader_id)):
                follower_indices[i] = [s for s in follower_indices[i] if s != torch.argmax(leader_id[i])]
            if self.args.cuda:
                inputs = inputs.cuda()
                inputs_next = inputs_next.cuda()
                self.election_hidden = self.election_hidden.cuda()
                self.target_election_hidden = self.target_election_hidden.cuda()
                self.eval_hidden = self.eval_hidden.cuda()
                self.eval_follower_hidden = self.eval_follower_hidden.cuda()
                self.target_hidden = self.target_hidden.cuda()
                self.target_follower_hidden = self.target_follower_hidden.cuda()
            inputs_l = torch.vstack([torch.mm(leader_id[j].float(), inputs[j]) for j in range(len(leader_id))]).unsqueeze(1).permute(1,0,2)
            q_eval, eval_hidden = self.eval_rnn(inputs_l, self.eval_hidden)  # inputs维度为(40,96)，得到的q_eval维度为(40,n_actions) for the leader
            self.eval_hidden = eval_hidden.reshape(episode_num, -1)
            inputs = inputs.numpy()
            inputs_f = torch.tensor(np.array([inputs[j, follower_indices[j], :] for j in range(len(follower_indices))]),
                                    dtype=torch.float32).permute(1, 0, 2)
            q_eval_f, eval_follower_hidden = self.eval_follower(inputs_f, q_eval, self.eval_follower_hidden)
            self.eval_follower_hidden = eval_follower_hidden.reshape(episode_num, -1)

            inputs_next_l = torch.vstack(
                [torch.mm(leader_id[j].float(), inputs_next[j]) for j in range(len(leader_id))]).unsqueeze(1).permute(1, 0, 2)
            # inputs_next_l = torch.tensor([inputs_next[:, [j], :] for j in leader_id][0].numpy(),
            #                              dtype=torch.float32).permute(1, 0, 2)
            q_target, target_hidden = self.target_rnn(inputs_next_l, self.target_hidden)
            self.target_hidden = target_hidden.reshape(episode_num, -1)

            inputs_next = inputs_next.numpy()
            inputs_f_next = torch.tensor(
                np.array([inputs_next[j, follower_indices[j], :] for j in range(len(follower_indices))]),
                dtype=torch.float32).permute(1, 0, 2)
            q_target_f, target_follower_hidden = self.target_follower(inputs_f_next, q_target,
                                                                      self.target_follower_hidden)
            self.target_follower_hidden = target_follower_hidden.reshape(episode_num, -1)

            # concat along the dimension of num_agents
            q_eval = torch.cat((q_eval, q_eval_f), dim=0)
            q_target = torch.cat((q_target, q_target_f), dim=0)

            # 把q_eval维度重新变回(8, 5,n_actions)
            q_eval = q_eval.view(episode_num, self.n_agents, -1)
            q_target = q_target.view(episode_num, self.n_agents, -1)
            q_evals.append(q_eval)
            q_targets.append(q_target)
            # 得的q_eval和q_target是一个列表，列表里装着max_episode_len个数组，数组的的维度是(episode个数, n_agents，n_actions)
            # 把该列表转化成(episode个数, max_episode_len， n_agents，n_actions)的数组



        q_evals = torch.stack(q_evals, dim=1)
        q_targets = torch.stack(q_targets, dim=1)
        return q_evals, q_targets

    def get_leader_id(self, inputs, election_hidden):
        '''

        :param inputs: [episode_num, num_agents, obs_shape]
        :param election_hidden: [episode_num, num_agents, rnn_hidden_dim]
        :return:
        '''
        inputs = inputs.reshape(-1, inputs.shape[-1])
        election_hidden = election_hidden.reshape(-1, election_hidden.shape[-1])
        eva, new_hidden = self.election(inputs, election_hidden)

        # inputs = torch.tensor(inputs)
        # with SummaryWriter(comment='rnnNet') as w:
        #     # w.add_graph(self.eval_rnn, (inputs_l, self.eval_hidden))
        #     w.add_graph(self.election, (inputs, election_hidden))

        # leader_id = int(torch.argmax(eva, dim=0))
        leader_id_one_hot = gumbel_softmax(eva.unsqueeze(0), hard=True)

        return leader_id_one_hot, new_hidden

    def init_hidden(self, episode_num):
        # 为每个episode中的每个agent都初始化一个eval_hidden、target_hidden
        self.election_hidden = torch.zeros((episode_num, self.args.n_agents, self.args.rnn_hidden_dim))
        self.target_election_hidden = torch.zeros((episode_num, self.args.n_agents, self.args.rnn_hidden_dim))
        self.eval_hidden = torch.zeros((episode_num,  self.args.rnn_hidden_dim))
        self.target_hidden = torch.zeros((episode_num, self.args.rnn_hidden_dim))
        self.eval_follower_hidden = torch.zeros((episode_num, self.args.rnn_hidden_dim))
        self.target_follower_hidden = torch.zeros((episode_num, self.args.rnn_hidden_dim))

    def save_model(self, train_step):
        num = str(train_step // self.args.save_cycle)
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        torch.save(self.eval_vdn_net.state_dict(), self.model_dir + '/' + num + '_qmix_net_params.pkl')
        torch.save(self.eval_rnn.state_dict(), self.model_dir + '/' + num + '_rnn_net_params.pkl')
