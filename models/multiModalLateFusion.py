import torch
import torch.nn as nn
from models.FCModel import FCClassifier
import torch.nn.functional as F


class LateFusionParClassifier(torch.nn.Module):

    def __init__(self, n_classes):
        super(LateFusionParClassifier, self).__init__()
        self.alpha = nn.Parameter(torch.tensor(0.5))
        self.netRGB = FCClassifier(n_classes=n_classes, modality='RGB')
        self.netEMG = FCClassifier(n_classes=n_classes, modality='EMG')

    def forward(self, data):

        # classifications network used to extract the logits (predictions)
        # self.alpha = nn.Parameter(torch.sigmoid(self.alpha))
        logitsRGB = self.netRGB(data)
        logitsEMG = self.netEMG(data)
        logits = self.alpha*logitsRGB + (1-self.alpha)*logitsEMG

        return logits,self.alpha.data


