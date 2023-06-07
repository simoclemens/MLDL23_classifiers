import torch
import torch.nn as nn
from models.FusionModel import FusionModel


class FCClassifier(torch.nn.Module):

    def __init__(self, n_classes, modality, batch_size=32, n_feat=1024):
        super(FCClassifier, self).__init__()

        self.classifier = torch.nn.Linear(1024, n_classes)
        self.modality = modality
        self.batch_size = batch_size
        self.n_feat = n_feat

        self.norm = nn.BatchNorm1d(self.n_feat)

    def forward(self, data):
        features = data[self.modality]

        norm_data = self.norm(features)

        # classifications network used to extract the logits (predictions)
        logits = self.classifier(norm_data)

        return logits