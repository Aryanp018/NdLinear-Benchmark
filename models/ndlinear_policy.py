import torch
import torch.nn as nn
from models.local_ndlinear import NdLinear

class NdLinearPolicy(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(NdLinearPolicy, self).__init__()
        self.model = nn.Sequential(
            NdLinear((input_dim,), (hidden_dim,)),
            nn.ReLU(),
            NdLinear((hidden_dim,), (output_dim,)),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        return self.model(x)
