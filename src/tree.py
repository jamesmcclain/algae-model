from typing import List

import numpy as np

import torch
import torch.hub
import torchvision as tv
import torch.nn.functional as F


class GreensFunctionLikeActivation(torch.nn.Module):
    def __init__(self, x0: float = 0.5, eps: float = 1e-3):
        super().__init__()
        a = torch.from_numpy(np.array([x0, eps]).astype(np.float))
        a = torch.nn.parameter.Parameter(a)
        self.register_parameter('a', a)

    def forward(self, x):
        out = -((x - self.a[0]) * (x - self.a[0])) / (self.a[1] * self.a[1])
        out = torch.exp(out)
        return out


class TreeModel(torch.nn.Module):
    def __init__(self, preshrink: int):
        super().__init__()

        self.cheaplab1 = torch.hub.load(
            'jamesmcclain/CheapLab:38af8e6cd084fc61792f29189158919c69d58c6a',
            'make_cheaplab_model',
            num_channels=224,
            preshrink=preshrink,
            out_channels=1)
        self.green11 = GreensFunctionLikeActivation()
        self.green12 = GreensFunctionLikeActivation()

        self.cheaplab2 = torch.hub.load(
            'jamesmcclain/CheapLab:38af8e6cd084fc61792f29189158919c69d58c6a',
            'make_cheaplab_model',
            num_channels=224,
            preshrink=preshrink,
            out_channels=1)
        self.green21 = GreensFunctionLikeActivation()
        self.green22 = GreensFunctionLikeActivation()

        self.cheaplab3 = torch.hub.load(
            'jamesmcclain/CheapLab:38af8e6cd084fc61792f29189158919c69d58c6a',
            'make_cheaplab_model',
            num_channels=224,
            preshrink=preshrink,
            out_channels=1)
        self.green31 = GreensFunctionLikeActivation()
        self.green32 = GreensFunctionLikeActivation()

        self.conv2d = torch.nn.Conv2d(9, 3, 1)

    def forward(self, x):
        x[x < 0] = 0

        x1 = self.cheaplab1(x)
        x11 = torch.sigmoid(self.green11(x1))
        x12 = torch.sigmoid(self.green12(x1))
        x2 = self.cheaplab2(x)
        x21 = torch.sigmoid(self.green21(x2))
        x22 = torch.sigmoid(self.green22(x2))
        x3 = self.cheaplab3(x)
        x31 = torch.sigmoid(self.green31(x3))
        x32 = torch.sigmoid(self.green32(x3))

        out = [
            x11, x11*x12, x12,
            x21, x21*x22, x22,
            x31, x31*x32, x32,
        ]
        out = torch.cat(out, dim=1)
        out = self.conv2d(out)

        return out


def make_tree_model(preshrink: int = 8):
    model = TreeModel(preshrink=preshrink)
    return model
