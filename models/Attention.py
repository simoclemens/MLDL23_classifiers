import torch
import torch.nn as nn
import torch.nn.functional as F
import math


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


class DotAttention(nn.Module):
    def __init__(self, input_size, num_clips, topk, d_q=512, d_k=512, d_v=512,device='cuda:0'):
        super(DotAttention, self).__init__()
        self.input_size = input_size
        self.num_clips = num_clips
        self.topk = topk
        self.linear = nn.Linear(input_size, 1)
        self.d_q = d_q
        self.d_k = d_k
        self.d_v = d_v
        self.device = device

        self.W_query = torch.nn.Parameter(torch.rand(self.d_q, input_size)
                                          / torch.sqrt(torch.tensor(2.0 / (self.d_q + input_size))))
        self.W_key = torch.nn.Parameter(torch.rand(self.d_k, input_size)
                                        / torch.sqrt(torch.tensor(2.0 / (self.d_k + input_size))))
        self.W_value = torch.nn.Parameter(torch.rand(self.d_v, input_size)
                                          / torch.sqrt(torch.tensor(2.0 / (self.d_v + input_size))))

        self.scale = torch.nn.Parameter(torch.rand(1))
        self.bias = torch.nn.Parameter(torch.rand(1))

    def forward(self, inputs):

        batch_size = inputs.shape[0]

        inputs_reshaped = inputs.view(batch_size * self.num_clips, self.input_size)

        # Apply batch normalization
        batch_norm = nn.BatchNorm1d(num_features=self.input_size).to(self.device)
        inputs_reshaped = batch_norm(inputs_reshaped).to(self.device)

        W_query_n = F.normalize(self.W_query)
        W_key_n = F.normalize(self.W_key)
        W_value_n = F.normalize(self.W_value)

        Q = W_query_n.matmul(inputs_reshaped.T).view(batch_size, self.num_clips, self.d_q)
        K = W_key_n.matmul(inputs_reshaped.T).view(batch_size, self.num_clips, self.d_k)
        V = W_value_n.matmul(inputs_reshaped.T).view(batch_size, self.num_clips, self.d_v)
        # print(self.W_value)

        omega = torch.bmm(Q, K.transpose(1, 2))
        logits = omega / (self.d_k ** 0.5)

        mask = torch.triu(torch.ones( self.num_clips), diagonal=1).unsqueeze(0)
        mask = mask.expand(batch_size, -1, -1).bool().to(self.device)

        logits = logits.masked_fill(mask, float('-inf'))

        attention_weights = torch.softmax(logits, dim=2)

        context_vector = torch.bmm(attention_weights, V)

        #concat_vector = context_vector.view(batch_size, self.num_clips * self.d_v)

        return context_vector[:,self.num_clips-1,:]


'''
 batch_size = inputs.shape[0]

        Q = torch.zeros((batch_size, self.num_clips, self.d_q)).to(self.device)
        K = torch.zeros((batch_size, self.num_clips, self.d_k)).to(self.device)
        V = torch.zeros((batch_size, self.num_clips, self.d_v)).to(self.device)

        attention_weights = torch.zeros((batch_size, self.num_clips, self.num_clips)).to(self.device)
        context_vector = torch.zeros((batch_size, self.num_clips, self.d_v)).to(self.device)

        for sample in range(batch_size):
            for clip in range(self.num_clips):
                Q[sample][clip] = self.W_query.matmul(inputs[sample][clip])
                K[sample][clip] = self.W_key.matmul(inputs[sample][clip])
                V[sample][clip] = self.W_value.matmul(inputs[sample][clip])

        for sample in range(batch_size):
            for clip in range(self.num_clips):
                omega = Q[sample][clip].matmul(K[sample].T)
                attention_weights = F.softmax(omega / self.d_k**0.5, dim=0)
                context_vector[sample][clip] = attention_weights.matmul(V[sample])
'''

class AttentionScore(nn.Module):
    def __init__(self, input_size=1024, num_clips=5,device='cuda:0'):
        super(AttentionScore, self).__init__()
        self.input_size = input_size
        self.num_clips = num_clips
        self.input_size = input_size
        self.device = device
        self.temp = 1.25
        self.bias = nn.Parameter(torch.randn(5))

        self.K_w = nn.Linear(self.input_size, self.input_size, bias=False)
        self.Q_w = nn.Linear(self.input_size, self.input_size, bias=False)

    def forward(self, inputs):

        batch_size = inputs.shape[0]

        inputs_reshaped = inputs.view(batch_size * self.num_clips, self.input_size)

        # Apply batch normalization
        batch_norm = nn.BatchNorm1d(num_features=self.input_size).to(self.device)
        inputs_reshaped = batch_norm(inputs_reshaped).to(self.device)

        Q = self.K_w(inputs_reshaped).view(batch_size, self.num_clips, self.input_size)
        K = self.Q_w(inputs_reshaped).view(batch_size, self.num_clips, self.input_size)

        omega = torch.bmm(Q, K.transpose(1, 2))
        logits = self.temp * (omega / (self.input_size ** 0.5))

        mask = torch.diag(torch.ones(5)).unsqueeze(0)
        mask = mask.expand(batch_size, -1, -1).bool().to(self.device)

        logits = logits.masked_fill(mask, float('-inf'))

        attention_weights = torch.softmax(logits, dim=2)
        attention_score = torch.mean(attention_weights,dim=1) #+ self.bias
        # attention_score = torch.softmax(attention_score, dim=1)

        return attention_score