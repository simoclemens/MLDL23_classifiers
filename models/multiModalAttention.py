import torch
from models.Attention import EnergyAttention
from models.FusionModel import FusionModel


class FusionClassifierAttention(torch.nn.Module):

    def __init__(self, topk, n_classes, batch_size, n_clips, hidden_size=512, device='cuda:0'):
        super(FusionClassifierAttention, self).__init__()

        self.topk = topk
        self.n_classes = n_classes
        self.batch_size = batch_size
        self.n_clips = n_clips
        self.device = device
        self.hidden_size = hidden_size

        self.net_fusion = FusionModel()
        self.classifier = torch.nn.Linear(self.hidden_size, self.n_classes)
        self.attention = EnergyAttention(input_size=self.hidden_size, num_clips=self.n_clips, topk=self.topk)

    def forward(self, data):
        image_features = data["RGB"]
        emg_features = data["EMG"]

        imageEMG_features = torch.zeros((self.batch_size, self.n_clips, self.hidden_size)).to(self.device)

        for clip in range(self.n_clips):
            # fusion layer
            imageEMG_features[:][clip] = self.net_fusion(image_features[:][clip], emg_features[:][clip])

        #imageEMG_features = imageEMG_features.transpose(0, 1)

        selected_features = self.attention(imageEMG_features)

        logits = torch.zeros((self.n_clips, self.batch_size, self.n_classes)).to(self.device)

        for clip in range(self.topk):

            # send all the data from the batch related to the given clip
            # inputs is a dictionary with key -> modality, value -> n rows related to the same clip
            inputs = selected_features[:, clip]

            output = self.classifier(inputs)  # get predictions from the net
            logits[clip] = output

        logits = torch.mean(logits, dim=0)

        return logits
