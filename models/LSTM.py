from torch import nn, relu
import torch.nn.functional as F
import torch
from torch.autograd import Variable
import numpy as np


class LSTM(nn.Module):
    def __init__(self, modality, n_clips=5, batch_size=32,input_size=1024, hidden_size=512, n_classes=20,
                 device='cuda:0',weights=None,weights_flag=False):
        super(LSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.n_classes = n_classes
        self.n_clips = n_clips
        self.modality = modality
        self.device = device
        self.weights=weights
        self.weights_flag=weights_flag
        self.lstm = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size, batch_first=True)
        self.classifier = nn.Linear(self.hidden_size, self.n_classes)
        self.batch_norm = nn.BatchNorm1d(num_features=self.input_size)
        self.batch_norm2 = nn.BatchNorm1d(num_features=self.hidden_size)

    def forward(self, data):
        features = data[self.modality].float()

        inputs_reshaped = features.view(self.batch_size * self.n_clips, self.input_size)
        features = self.batch_norm(inputs_reshaped).view(self.batch_size, self.n_clips, self.input_size)

        logits = torch.zeros((self.n_clips, self.batch_size, self.n_classes)).to(self.device)

        # Initialize the hidden state and cell state
        hidden_state = torch.zeros(1, self.batch_size, self.hidden_size).to(self.device)
        cell_state = torch.zeros(1, self.batch_size, self.hidden_size).to(self.device)
        hidden = (hidden_state, cell_state)

        for clip in range(self.n_clips):
            hidden = (self.batch_norm2(hidden[0].squeeze(0)).unsqueeze(0),
                      self.batch_norm2(hidden[1].squeeze(0)).unsqueeze(0))
            # Get the current clip's input
            clip_in = features[:, clip, :]
            # Pass the input through the LSTM
            output, hidden = self.lstm(clip_in.unsqueeze(1), hidden)  # Unsqueeze to add a time step dimension
            # Save the output of the clip
            logits[clip] = self.classifier(output.squeeze(1))  # Squeeze to remove the time step dimension
        if self.weights_flag == False :
            logits = torch.mean(logits, dim=0)

        return logits
