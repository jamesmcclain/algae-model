from typing import List

import torch
import torch.hub
import torchvision as tv
import torch.nn.functional as F


class Nugget(torch.nn.Module):
    def __init__(self, kernel_size, in_channels, out_channels):
        super(Nugget, self).__init__()
        self.conv2ds = torch.nn.ModuleDict()
        for n in in_channels:
            self.conv2ds[str(n)] = torch.nn.Conv2d(n, 10, kernel_size=kernel_size)
        self.batch_norm = torch.nn.BatchNorm2d(10)
        self.relu = torch.nn.ReLU()
        self.cheaplab = torch.hub.load(
            'jamesmcclain/CheapLab:38af8e6cd084fc61792f29189158919c69d58c6a',
            'make_cheaplab_model',
            num_channels=10,
            preshrink=8,
            out_channels=out_channels)

    def forward(self, x):
        n = x.shape[-3]
        out = self.conv2ds[str(n)](x)
        out = self.batch_norm(out)
        out = self.relu(out)
        out = self.cheaplab(out)
        return out


class CloudModel(torch.nn.Module):
    def __init__(self, in_channels: List[int]):
        super().__init__()
        self.rs = torch.nn.ModuleList([Nugget(1, in_channels, 1) for i in range(3)])
        self.gs = torch.nn.ModuleList([Nugget(1, in_channels, 1) for i in range(3)])
        self.bgs = torch.nn.ModuleList([Nugget(1, in_channels, 1) for i in range(3)])

    def forward(self, x):
        x[x < 0] = 0
        F.normalize(x, dim=1)

        rs = [m(x) for m in self.rs]
        rs = torch.cat(rs, dim=1)
        F.normalize(rs, dim=1)

        gs = [m(x) for m in self.gs]
        gs = torch.cat(gs, dim=1)
        F.normalize(gs, dim=1)

        bgs = [m(x) for m in self.bgs]
        bgs = torch.cat(bgs, dim=1)
        F.normalize(bgs, dim=1)

        out = [
            torch.unsqueeze(torch.amax(rs, dim=1), dim=1),
            torch.unsqueeze(torch.amax(gs, dim=1), dim=1),
            torch.unsqueeze(torch.amax(bgs, dim=1), dim=1)
        ]
        out = torch.cat(out, dim=1)
        F.normalize(out, dim=1)
        goodness = -torch.mean(torch.std(rs, dim=1, unbiased=True)) \
            - torch.mean(torch.std(gs, dim=1, unbiased=True)) \
            - torch.mean(torch.std(bgs, dim=1, unbiased=True))

        return (out, goodness)


def make_cloud_model(in_channels: List[int]):
    model = CloudModel(in_channels=in_channels)
    return model
