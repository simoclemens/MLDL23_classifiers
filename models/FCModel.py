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

        #self.fc1 = torch.nn.Linear(self.n_feat, int(self.n_feat/2))
        #self.fc2 = torch.nn.Linear(int(self.n_feat/2), n_classes)
        self.classifier = torch.nn.Linear(self.n_feat, self.n_classes,bias=True)
        torch.nn.init.xavier_uniform_(self.classifier.weight)
        #torch.nn.init.xavier_uniform_(self.classifier.bias)
        #self.relu = nn.ReLU()
        self.norm = nn.BatchNorm1d(self.n_feat)

    def forward(self, data):
        features = data[self.modality]

        norm_data = self.norm(features)

        # classifications network used to extract the logits (predictions)
        # out = self.fc1(norm_data)
        # out = self.relu(out)
        # out = self.fc2(out)
        out = self.classifier(norm_data)
        return out