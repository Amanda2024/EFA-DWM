import torch
import torch.nn as nn
import torch.nn.functional as f

class Election_Model(nn.Module):
    # Select a Leader from many agents based on the history of each agent
    def __init__(self, input_shape, args):
        super(Election_Model, self).__init__()
        self.args = args

        self.fc1 = nn.Linear(input_shape, args.rnn_hidden_dim)
        self.dropout = nn.Dropout(p=0.2)
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.att = nn.MultiheadAttention(embed_dim=args.rnn_hidden_dim, num_heads=args.num_heads)
        self.fc2 = nn.Linear(args.rnn_hidden_dim, 1)

    def forward(self, taus, hidden_state):
        '''
        :param taus: history of all agents [num_agents, input_shape]
        :param hidden_state: hidden state of last timestep []
        :return: eva, h
        '''
        x = f.relu(self.fc1(taus))
        drop_1 = self.dropout(x)
        h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
        h = self.rnn(drop_1, h_in).unsqueeze(0)
        # drop_2 = self.dropout(h)
        att_out, _ = self.att(h, h, h)
        #print('heads:', self.args.num_heads)
        eva = self.fc2(att_out).reshape(self.args.n_agents)
        # drop_3 = self.dropout(eva)
        eva = f.softmax(eva)
        return eva, h



