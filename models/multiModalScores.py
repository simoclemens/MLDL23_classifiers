import torch
from models.Attention import EnergyAttention
from models.Attention import AttentionScore
from models.FusionModel import FusionModel
import torch.nn as nn


class FusionClassifierScore(torch.nn.Module):
    def __init__(self, topk, n_classes, batch_size, n_clips, hidden_size=512, device='cuda:0'):
        super(FusionClassifierScore, self).__init__()

        self.topk = topk
        self.n_classes = n_classes
        self.batch_size = batch_size
        self.n_clips = n_clips
        self.device = device
        self.hidden_size = hidden_size

        self.net_fusion = FusionModel()
        self.fc1 = torch.nn.Linear(self.hidden_size, n_classes)
        self.relu = torch.nn.ReLU()
        self.attention = AttentionScore(input_size=self.hidden_size, num_clips=self.n_clips, topk=self.topk)

    def forward(self, data):
        image_features = data["RGB"]
        emg_features = data["EMG"]

        imageEMG_features = torch.zeros((self.batch_size, self.n_clips, self.hidden_size)).to(self.device)

        logits = torch.zeros((self.n_clips, self.batch_size,  self.n_classes)).to(self.device)

        for clip in range(self.n_clips):
            # fusion layer
            imageEMG_features[:][clip] = self.net_fusion(image_features[:][clip], emg_features[:][clip])


        attention_scores = self.attention(imageEMG_features)

        for clip in range(self.n_clips):
                output = self.fc1(imageEMG_features[:, clip])*attention_scores[clip].unsqueeze(1)
                logits[clip] = output

        logits = torch.mean(logits, dim=0)

        return logits
