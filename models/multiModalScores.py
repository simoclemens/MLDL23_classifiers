import torch
from models.Attention import EnergyAttention
from models.Attention import AttentionScore
from models.FusionModel import FusionModel
from models.FCModel import FCClassifier
import torch.nn as nn


class ScoreClassifier(torch.nn.Module):
    def __init__(self, n_classes, f1, f2, batch_size=32, n_clips=5, hidden_size=512, device='cuda:0'):
        super(ScoreClassifier, self).__init__()

        self.n_classes = n_classes
        self.batch_size = batch_size
        self.n_clips = n_clips
        self.device = device
        self.hidden_size = hidden_size
        self.f1 = f1
        self.f2 = f2

        self.net_fusion = FusionModel()
        self.classifier = FCClassifier(n_classes=self.n_classes,modality=self.f1)
        self.relu = torch.nn.ReLU()
        self.attention = AttentionScore()

    def forward(self, data):
        emg_features = data[self.f2]
        rgb_features = {}

        logits = torch.zeros((self.n_clips, self.batch_size, self.n_classes)).to(self.device)

        attention_scores = self.attention(emg_features)

        for clip in range(self.n_clips):
            rgb_features[self.f1] = data[self.f1][:, clip]
            output = self.classifier(rgb_features)*attention_scores[:, clip].unsqueeze(1)
            logits[clip] = output

        logits = torch.mean(logits, dim=0)

        return logits, attention_scores
