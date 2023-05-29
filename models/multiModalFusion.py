import torch
import torch.nn
from models.FusionModel import FusionModel


class FusionClassifier(torch.nn.Module):

    def __init__(self, n_classes):
        super(FusionClassifier, self).__init__()

        self.net_fusion = FusionModel()
        self.classifier = torch.nn.Linear(512, n_classes)

    def forward(self, data):
        image_features = data["RGB"]
        audio_features = data["EMG"]


        # fusion layer
        imageAudio_features = self.net_fusion(image_features, audio_features)

        # classifications network used to extract the logits (predictions)
        logits = self.classifier(imageAudio_features)

        return logits


