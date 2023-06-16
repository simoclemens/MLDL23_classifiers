import torch
import torch.nn as nn


class FusionModel(torch.nn.Module):

    # this code defines a simple model for jointly embedding image and audio features

    def __init__(self):
        super(FusionModel, self).__init__()
        #initialize model
        self.imageAudio_fc1 = torch.nn.Linear(1024 * 2, 512 * 2)
        self.imageAudio_fc2 = torch.nn.Linear(512 * 2, 512)
        self.relu = torch.nn.ReLU()
        self.norm = nn.BatchNorm1d(2048)
        self.norm0 = nn.BatchNorm1d(1024)

    def forward(self, image_features, audio_features):
        image_features = self.norm0(image_features)
        audio_features = self.norm0(audio_features)
        audioVisual_features = torch.cat((image_features, audio_features), dim=1)
        imageAudio_embedding = audioVisual_features
        #imageAudio_embedding = self.imageAudio_fc1(audioVisual_features)
        #imageAudio_embedding = self.relu(imageAudio_embedding)
        #imageAudio_embedding = self.imageAudio_fc2(imageAudio_embedding)
        output = self.norm(imageAudio_embedding)
        return output
