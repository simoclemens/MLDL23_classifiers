import torch
import torch.nn


class FusionClassifier(torch.nn.Module):

    def __init__(self, n_classes):
        super(FusionClassifier, self).__init__()

        self.net_imageAudio = ImageAudioModel()
        self.classifier = torch.nn.Linear(512, n_classes)

    def forward(self, data):
        image_features = data["RGB"]
        audio_features = data["EMG"]


        # fusion layer
        imageAudio_features = self.net_imageAudio(image_features, audio_features)

        # classifications network used to extract the logits (predictions)
        logits = self.classifier(imageAudio_features)

        return logits


class ImageAudioModel(torch.nn.Module):

    # this code defines a simple model for jointly embedding image and audio features

    def __init__(self):
        super(ImageAudioModel, self).__init__()
        #initialize model
        self.imageAudio_fc1 = torch.nn.Linear(1024 * 2, 512 * 2)
        self.imageAudio_fc2 = torch.nn.Linear(512 * 2, 512)
        self.relu = torch.nn.ReLU()

    def forward(self, image_features, audio_features):
        audioVisual_features = torch.cat((image_features, audio_features), dim=1)
        imageAudio_embedding = self.imageAudio_fc1(audioVisual_features)
        imageAudio_embedding = self.relu(imageAudio_embedding)
        imageAudio_embedding = self.imageAudio_fc2(imageAudio_embedding)
        return imageAudio_embedding
