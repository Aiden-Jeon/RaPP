import torch
import torch.nn as nn


class FullyConnectedLayer(nn.Module):
    def __init__(self, input_size: int, output_size: int, act: str):
        super().__init__()
        layer = [nn.Linear(input_size, output_size), nn.BatchNorm1d(output_size)]
        if act == "leakyrelu":
            layer += [nn.LeakyReLU()]
        elif act == "sigmoid":
            layer += [nn.Sigmoid()]
        self.layer = nn.Sequential(*layer)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.layer(X)
