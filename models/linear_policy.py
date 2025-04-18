import torch
import torch.nn as nn

class LinearPolicy(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(LinearPolicy, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        return self.model(x)
