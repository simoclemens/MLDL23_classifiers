import torch
import torch.nn as nn
from models.FusionModel import FusionModel


class FusionClassifier(torch.nn.Module):

    def __init__(self, n_classes):
        super(FusionClassifier, self).__init__()

        self.net_fusion = FusionModel()
        self.classifier1 = torch.nn.Linear(2048, 1024)
        self.classifier2 = torch.nn.Linear(1024, n_classes)

    def forward(self, data):
        image_features = data["RGB"]
        audio_features = data["EMG"]

        # fusion layer
        imageAudio_features = self.net_fusion(image_features, audio_features)

        # classifications network used to extract the logits (predictions)
        logits = self.classifier1(imageAudio_features)
        logits = self.classifier2(logits)

        return logits


