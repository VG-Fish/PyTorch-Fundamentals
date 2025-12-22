import torch

# 10
torch.manual_seed(7)
T = torch.rand(1, 1, 1, 10)
sT = T.squeeze()
print(T, T.shape)
print(sT, sT.shape)
