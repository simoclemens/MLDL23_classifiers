import torch
import torch.nn as nn
from models.FusionModel import FusionModel


class FCClassifier(torch.nn.Module):

    def __init__(self, n_classes, modality, batch_size=32, n_feat=1024):
        super(FCClassifier, self).__init__()
        self.modality = modality
        self.batch_size = batch_size
        self.n_feat = n_feat
        self.n_classes = n_classes

        self.classifier = torch.nn.Linear(self.n_feat, self.n_classes,bias=True)
        torch.nn.init.xavier_uniform_(self.classifier.weight)
        self.norm = nn.BatchNorm1d(self.n_feat)

        self.a = nn.Parameter(torch.randn(1))
        self.b = nn.Parameter(torch.randn(1))

    def forward(self, data):
        features = data[self.modality]
        norm_data = self.norm(features)

        out = self.classifier(norm_data)
        return out