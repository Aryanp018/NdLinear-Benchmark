import torch
import torch.nn as nn

class NdLinear(nn.Module):
    def __init__(self, input_dims: tuple, hidden_size: tuple, transform_outer=True):
        super(NdLinear, self).__init__()

        if len(input_dims) != len(hidden_size):
            raise Exception("Input shape and hidden shape do not match.")

        self.input_dims = input_dims
        self.hidden_size = hidden_size
        self.num_layers = len(input_dims)
        self.transform_outer = transform_outer

        self.align_layers = nn.ModuleList([
            nn.Linear(input_dims[i], hidden_size[i]) for i in range(self.num_layers)
        ])

    def forward(self, X):
        num_transforms = self.num_layers

        for i in range(num_transforms):
            if self.transform_outer:
                layer = self.align_layers[i]
                transpose_dim = i + 1
            else:
                layer = self.align_layers[num_transforms - (i + 1)]
                transpose_dim = num_transforms - i

            X = torch.transpose(X, transpose_dim, num_transforms).contiguous()
            X_size = X.shape[:-1]
            X = X.view(-1, X.shape[-1])
            X = layer(X)
            X = X.view(*X_size, X.shape[-1])
            X = torch.transpose(X, transpose_dim, num_transforms).contiguous()

        return X
