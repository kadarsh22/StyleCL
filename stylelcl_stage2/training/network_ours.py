import torch
import torch.nn as nn
from torch_utils import persistence
import math


@persistence.persistent_class
class Adaptor(nn.Module):
    def __init__(self, num_basis_vec=16, w_dim=512, num_layers=14):
        super().__init__()
        self.num_layers = num_layers
        self.adaptor = nn.ModuleList([nn.Linear(num_basis_vec, w_dim) for _ in range(num_layers)])
        offset_tensor = torch.nn.init.kaiming_uniform_(torch.zeros(num_layers, w_dim), a=math.sqrt(5))
        self.offset = nn.Parameter(offset_tensor)

    def forward(self, x):
        mean_vec = torch.stack([self.adaptor[i](x[:, i, :]) for i in range(self.num_layers)]).permute(1, 0, 2)
        return mean_vec + self.offset
