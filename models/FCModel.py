import torch
import torch.nn
from models.FusionModel import FusionModel


class FCClassifier(torch.nn.Module):

    def __init__(self, n_classes):
        super(FCClassifier, self).__init__()

        self.classifier = torch.nn.Linear(1024, n_classes)

    def forward(self, data):
        image_features = data["RGB"]

        # classifications network used to extract the logits (predictions)
        logits = self.classifier(image_features)

        return logits