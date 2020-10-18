"""
/*******************************************
* Custom LSTM (Butterfly, Circulant, Standard)
********************************************/
"""

import torch.nn as nn
import torch
import torch.nn.functional as F
from collections import OrderedDict
import math
from torch_butterfly.butterfly import Butterfly

device = torch.device("cuda")


class LSTMCell(nn.Module):

    def __init__(self, input_size, hidden_size, bias=True, butterfly=False):
        super(LSTMCell, self).__init__()
        self.input_size= input_size
        self.hidden_size = hidden_size
        self.bias = bias
        if butterfly:
            self.Gates = Butterfly(input_size+hidden_size, 4*hidden_size, bias=bias, complex=False, init='ortho', nblocks=2)
        else:
            self.Gates = nn.Linear(input_size+hidden_size, 4*hidden_size, bias=bias)
            self.reset_parameters()


    def forward(self, x, hidden):

        h_cur, c_cur = hidden
        h_cur = h_cur.to(device)
        c_cur = c_cur.to(device)

        combined = torch.cat([x, h_cur], dim=1)

        gates = self.Gates(combined)

        in_gate, remember_gate, out_gate, cell_gate = gates.chunk(4, 1)

        in_gate = self.hard_sigmoid(in_gate)
        remember_gate = self.hard_sigmoid(remember_gate)
        out_gate = self.hard_sigmoid(out_gate)
        cell_gate = torch.tanh(cell_gate)

        cell = (remember_gate * c_cur) + (in_gate * cell_gate)
        hidden = out_gate * torch.tanh(cell)

        return hidden, cell

    def init_hidden(self, batch_size):
        return (torch.zeros(batch_size, self.hidden_size, self.input_size),
                torch.zeros(batch_size, self.hidden_size, self.input_size))

    def hard_sigmoid(self, x):
        """
        Computes element-wise hard sigmoid of x.
        See e.g. https://github.com/Theano/Theano/blob/master/theano/tensor/nnet/sigm.py#L279
        """
        x = (0.2 * x) + 0.5
        x = F.threshold(-x, -1, -1)
        x = F.threshold(-x, 0, 0)

        return x

    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)


class LSTMLayer(nn.Module):
    def __init__(self, input_size, hidden_size, batch_first=True, bias=True, butterfly=False):
        super(LSTMLayer, self).__init__()
        # the input_shape is 5D tensor (batch, seq_length, channels, height, width)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.lstm= LSTMCell(input_size=input_size,hidden_size=hidden_size, bias=bias, butterfly=butterfly)

    def forward(self, x, hidden_state=None):

        if hidden_state is None:
            hidden_state = (
                torch.zeros(x.size(0), self.hidden_size),
                torch.zeros(x.size(0), self.hidden_size)
            )

        T = x.size(1) # seq_length
        h, c = hidden_state
        output_inner = []
        for t in range(T):
            h, c = self.lstm(x[:, t], hidden=[h, c])
            output_inner.append(h)
        layer_output = torch.stack(output_inner, dim=1)
        return [layer_output]


"""
class LSTM(nn.Module):

    def __init__(self, input_size, hidden_size, bias=True):
        super(LSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.i2h = nn.Linear(input_size, 4 * hidden_size, bias=bias)
        self.h2h = nn.Linear(hidden_size, 4 * hidden_size, bias=bias)
        self.reset_parameters()

    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)

    def forward(self, x, hidden):
        h, c = hidden
        h = h.view(h.size(1), -1)
        c = c.view(c.size(1), -1)
        x = x.view(x.size(1), -1)

        # Linear mappings
        preact = self.i2h(x) + self.h2h(h)

        # activations
        gates = preact[:, :3 * self.hidden_size].sigmoid()
        g_t = preact[:, 3 * self.hidden_size:].tanh()
        i_t = gates[:, :self.hidden_size]
        f_t = gates[:, self.hidden_size:2 * self.hidden_size]
        o_t = gates[:, -self.hidden_size:]

        c_t = th.mul(c, f_t) + th.mul(i_t, g_t)

        h_t = th.mul(o_t, c_t.tanh())

        h_t = h_t.view(1, h_t.size(0), -1)
        c_t = c_t.view(1, c_t.size(0), -1)
        return h_t, (h_t, c_t)
"""