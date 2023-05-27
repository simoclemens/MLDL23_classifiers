import torch


class FusionClassifier(torch.nn.Module):

    def __init__(self, net_imageAudio, net_classifier):
        super(FusionClassifier, self).__init__()

        self.net_imageAudio = net_imageAudio
        self.net_classifier = net_classifier

    def forward(self, data):
        image_features = data["RGB"]
        audio_features = data["EMG"]


        # fusion layer
        imageAudio_features = self.net_imageAudio(image_features, audio_features)

        # classifications network used to extract the logits (predictions)
        logits = self.net_classifier(imageAudio_features)

        return logits
