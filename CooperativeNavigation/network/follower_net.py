import torch.nn as nn
import torch.nn.functional as f
import torch


class Follower_RNN(nn.Module):
    # The follower must take as input the output of the leader agent, input_shape=obs_shape+n_actions+n_agents
    def __init__(self, input_shape, args):
        super(Follower_RNN, self).__init__()
        self.args = args

        self.fc1 = nn.Linear(input_shape, args.rnn_hidden_dim)
        self.dropout = nn.Dropout(p=0.2)
        # self.rnn = nn.GRUCell(args.rnn_hidden_dim + args.action_dim, args.rnn_hidden_dim)
        self.rnn = nn.RNN(args.rnn_hidden_dim + args.action_dim, args.rnn_hidden_dim, 1)
        self.fc2 = nn.Linear(args.rnn_hidden_dim, args.n_actions)

    def forward(self, obs, f_leader, hidden_state):
        x = f.relu(self.fc1(obs))
        x = self.dropout(x)
        f_leader = torch.repeat_interleave(f_leader, repeats=x.shape[0], dim=0)
        x = torch.cat([x, f_leader], dim=-1)
        h_in = hidden_state.reshape(1, -1, self.args.rnn_hidden_dim)
        y, h = self.rnn(x, h_in)
        q = self.fc2(y)
        return q, h
