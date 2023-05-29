from torch import nn, relu
import torch.nn.functional as F
import torch
from torch.autograd import Variable
import numpy as np


class GRU(nn.Module):
    def __init__(self, num_classes=8, input_size=1024, hidden_size=256, num_layers=1, batch_size = 32):
        super(GRU, self).__init__()
        self.num_classes = num_classes
        self.input_size = input_size
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.gru = nn.GRU(input_size=self.input_size, hidden_size=self.hidden_size, num_layers=self.num_layers)
        self.fc1 = nn.Linear(self.hidden_size, 128)
        self.fc2 = nn.Linear(128, self.num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        # h_0 = torch.zeros(self.num_layers, self.hidden_size)  # hidden state
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # h_0.to(device)
        # c_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size))  # internal state
        # output from lstm network
        out, hn = self.gru(x)  # lstm with input, hidden, and internal state
        out = out[:, -1, :]
        out = self.relu(out)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)

        return out
