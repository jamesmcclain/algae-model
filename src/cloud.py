from typing import List

import torch
import torch.hub
import torchvision as tv


class Nugget(torch.nn.Module):
    def __init__(self, kernel_size, in_channels, out_channels):
        super(Nugget, self).__init__()
        self.conv2d = torch.nn.Conv2d(in_channels,
                                      out_channels,
                                      kernel_size=kernel_size)
        self.batch_norm = torch.nn.BatchNorm2d(out_channels)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = self.conv2d(x)
        x = self.batch_norm(x)
        x = self.relu(x)
        return x


class CloudModel(torch.nn.Module):
    def __init__(self, in_channels: List[int]):
        super().__init__()
        self.in_nuggets = torch.nn.ModuleDict()
        for n in in_channels:
            self.in_nuggets[str(n)] = Nugget(
                kernel_size=1,
                in_channels=n,
                out_channels=64)
        self.cheaplab = torch.hub.load(
            'jamesmcclain/CheapLab:38af8e6cd084fc61792f29189158919c69d58c6a',
            'make_cheaplab_model',
            num_channels=64,
            out_channels=3)

    def forward(self, x):
        [w, h] = x.shape[-2:]
        n = x.shape[-3]
        out = x

        nugget = self.in_nuggets[str(n)]
        if nugget is None:
            raise Exception(f"no Nugget for {n} channels")
        out = nugget(out)
        out = self.cheaplab(out)

        return out


def make_cloud_model(in_channels: List[int]):
    model = CloudModel(in_channels=in_channels)
    return model
