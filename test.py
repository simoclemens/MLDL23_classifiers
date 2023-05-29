import torch

w = torch.rand(32,5,1024)

print(w[1][2].size())