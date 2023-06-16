import torch

mask = torch.diag(torch.ones(5)).unsqueeze(0)
print(mask)
mask = mask.expand(32, -1, -1).bool()
print(mask)
logits = logits.masked_fill(mask, 0)