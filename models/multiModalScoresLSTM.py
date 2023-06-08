import torch
from models.Attention import EnergyAttention
from models.Attention import AttentionScore
from models.FusionModel import FusionModel
from models.FCModel import FCClassifier
import torch.nn as nn


class ScoreClassifierLSTM(torch.nn.Module):
    def __init__(self, n_classes,f1,f2, batch_size=32, n_clips=5, hidden_size=512, input_size = 1024,device='cuda:0'):
        super(ScoreClassifierLSTM, self).__init__()

        self.n_classes = n_classes
        self.batch_size = batch_size
        self.n_clips = n_clips
        self.device = device
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.f1 = f1
        self.f2 = f2

        self.net_fusion = FusionModel()
        # self.classifier = FCClassifier(n_classes=self.n_classes,modality=self.f1)
        self.relu = torch.nn.ReLU()
        self.attention = AttentionScore()
        self.batch_norm = nn.BatchNorm1d(num_features=self.input_size)

        self.lstm = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size, batch_first=True)
        self.classifier = nn.Linear(self.hidden_size, self.n_classes)

    def forward(self, data):
        emg_features = data[self.f2]
        rgb_features = {}
        logits = torch.zeros((self.n_clips, self.batch_size, self.n_classes)).to(self.device)

        attention_scores = self.attention(emg_features)

        features = data[self.f1]

        inputs_reshaped = features.view(self.batch_size * self.n_clips, self.input_size)
        features = self.batch_norm(inputs_reshaped).view(self.batch_size, self.n_clips, self.input_size)


        # Initialize the hidden state and cell state
        hidden_state = torch.zeros(1, self.batch_size, self.hidden_size).to(self.device)
        cell_state = torch.zeros(1, self.batch_size, self.hidden_size).to(self.device)
        hidden = (hidden_state, cell_state)

        for clip in range(self.n_clips):
            clip_in = features[:, clip, :]
            output, hidden = self.lstm(clip_in.unsqueeze(1), hidden)
            logits[clip] = self.classifier(output.squeeze(1))*attention_scores[:, clip].unsqueeze(1)

        logits = torch.mean(logits, dim=0)

        return logits
