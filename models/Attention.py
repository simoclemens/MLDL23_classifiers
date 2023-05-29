import torch
import torch.nn as nn
import torch.nn.functional as F


class EnergyAttention(nn.Module):
    def __init__(self, input_size, num_clips, topk):
        super(EnergyAttention, self).__init__()
        self.input_size = input_size
        self.num_clips = num_clips
        self.topk = topk
        self.linear = nn.Linear(input_size, 1)

    def forward(self, inputs):
        # Apply linear transformation
        energy = self.linear(inputs)
        # energy: [batch_size, num_clips, 1]

        # Apply softmax to obtain attention weights
        attention_weights = F.softmax(energy, dim=1)
        # attention_weights: [batch_size, num_clips, 1]

        # Get top-k clips with highest attention weights
        _, topk_indices = torch.topk(attention_weights.squeeze(2), self.topk, dim=1)
        # topk_indices: [batch_size, topk]

        # Gather the top-k clips from the input tensor
        selected_clips = torch.gather(inputs, 1, topk_indices.unsqueeze(2).expand(-1, -1, self.input_size))
        # selected_clips: [batch_size, topk, input_size]

        return selected_clips

    class EnergyAttention(nn.Module):
        def __init__(self, input_size, num_clips, topk):
            super(EnergyAttention, self).__init__()
            self.input_size = input_size
            self.num_clips = num_clips
            self.topk = topk
            self.linear = nn.Linear(input_size, 1)

        def forward(self, inputs):
            # Apply linear transformation
            energy = self.linear(inputs)
            # energy: [batch_size, num_clips, 1]

            # Apply softmax to obtain attention weights
            attention_weights = F.softmax(energy, dim=1)
            # attention_weights: [batch_size, num_clips, 1]

            # Get top-k clips with highest attention weights
            _, topk_indices = torch.topk(attention_weights.squeeze(2), self.topk, dim=1)
            # topk_indices: [batch_size, topk]

            # Gather the top-k clips from the input tensor
            selected_clips = torch.gather(inputs, 1, topk_indices.unsqueeze(2).expand(-1, -1, self.input_size))
            # selected_clips: [batch_size, topk, input_size]

            return selected_clips

