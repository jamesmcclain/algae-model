#!/usr/bin/env python3

import argparse
import csv
import logging
import sys
import math

import numpy as np
import torch
import torch.hub
import torch.nn.functional as F
import torchvision as tv
import tqdm
from torch.utils.data import DataLoader
from torchvision.io import VideoReader
from torchvision.transforms.functional import F_t


def cli_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--backbone',
                        required=False,
                        type=str,
                        default='resnet18')
    parser.add_argument('--epochs1', required=False, type=int, default=33)
    parser.add_argument('--epochs2', required=False, type=int, default=72)
    parser.add_argument('--imagery',
                        required=False,
                        choices=['aviris', 'sentinel2'],
                        default='aviris')
    parser.add_argument('--pth-load', required=False, type=str, default=None)
    parser.add_argument('--pth-save', required=False, type=str, default=None)
    parser.add_argument('--savez', required=True, type=str)

    parser.add_argument('--cheaplab', dest='cheaplab', action='store_true')
    parser.add_argument('--no-cheaplab', dest='cheaplab', action='store_false')
    parser.set_defaults(cheaplab=True)

    return parser


dataloader_cfg = {
    'batch_size': 128,
    'num_workers': 8,
    'shuffle': True,
}


def freeze(m):
    for p in m.parameters():
        p.requires_grad = False


def unfreeze(m):
    for p in m.parameters():
        p.requires_grad = True


def greens_function(x, x0, eps=1e-3):
    out = -((x - x0) * (x - x0)) / (eps * eps)
    out = torch.exp(out)
    return out


class AlgaeClassifier(torch.nn.Module):
    def __init__(self,
                 imagery: str = 'aviris',
                 use_cheaplab: bool = True,
                 backbone_str: str = None):
        super().__init__()

        self.imagery = imagery
        self.use_cheaplab = use_cheaplab
        self.backbone_str = backbone_str

        # Number of input bands
        if self.imagery == 'aviris':
            n = 224
        elif self.imagery == 'sentinel2':
            n = 12
        else:
            raise Exception(f'unknown imagery type {self.imagery}')

        # Number of bands going into the backbone
        if self.use_cheaplab:
            m = 2
            self.cheaplab = torch.hub.load('jamesmcclain/CheapLab:08d260b',
                                           'make_cheaplab_model',
                                           num_channels=n)
        else:
            m = n

        backbone = getattr(tv.models, self.backbone_str)
        self.backbone = backbone(pretrained=True)

        # Input and output
        # if self.backbone_str == 'alexnet':
        #     self.first = self.backbone.features[0] = torch.nn.Conv2d(
        #         m, 64, kernel_size=(11, 11), stride=(4, 4), padding=(2, 2))
        #     self.last = self.backbone.classifier[6] = torch.nn.Linear(
        #         in_features=4096, out_features=1, bias=True)
        if self.backbone_str == 'vgg16':
            self.first = self.backbone.features[0] = torch.nn.Conv2d(
                m, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            self.last = self.backbone.classifier[6] = torch.nn.Linear(
                in_features=4096, out_features=1, bias=True)
        elif self.backbone_str == 'squeezenet1_0':
            self.first = self.backbone.features[0] = torch.nn.Conv2d(
                m, 96, kernel_size=(7, 7), stride=(2, 2))
            self.last = self.backbone.classifier[1] = torch.nn.Conv2d(
                512, 1, kernel_size=(1, 1), stride=(1, 1))
        elif self.backbone_str == 'densenet161':
            self.first = self.backbone.features.conv0 = \
                torch.nn.Conv2d(
                    m,
                    96,
                    kernel_size=(7, 7),
                    stride=(2, 2),
                    padding=(3, 3),
                    bias=False)
            self.last = self.backbone.classifier = torch.nn.Linear(
                in_features=2208, out_features=1, bias=True)
        elif self.backbone_str == 'shufflenet_v2_x1_0':
            self.first = self.backbone.conv1[0] = torch.nn.Conv2d(
                m,
                24,
                kernel_size=(3, 3),
                stride=(2, 2),
                padding=(1, 1),
                bias=False)
            self.last = self.backbone.fc = torch.nn.Linear(in_features=1024,
                                                           out_features=1,
                                                           bias=True)
        elif self.backbone_str == 'mobilenet_v2':
            self.first = self.backbone.features[0][0] = torch.nn.Conv2d(
                m,
                32,
                kernel_size=(3, 3),
                stride=(2, 2),
                padding=(1, 1),
                bias=False)
            self.last = self.backbone.classifier[1] = torch.nn.Linear(
                in_features=1280, out_features=1, bias=True)
        elif self.backbone_str in ['mobilenet_v3_large', 'mobilenet_v3_small']:
            self.first = self.backbone.features[0][0] = torch.nn.Conv2d(
                m,
                16,
                kernel_size=(3, 3),
                stride=(2, 2),
                padding=(1, 1),
                bias=False)
            in_features = self.backbone.classifier[0].out_features
            self.last = self.backbone.classifier[3] = torch.nn.Linear(
                in_features=in_features, out_features=1, bias=True)
        elif self.backbone_str == 'mnasnet1_0':
            self.first = self.backbone.layers[0] = torch.nn.Conv2d(
                m,
                32,
                kernel_size=(3, 3),
                stride=(2, 2),
                padding=(1, 1),
                bias=False)
            self.last = self.backbone.classifier[1] = torch.nn.Linear(
                in_features=1280, out_features=1, bias=True)
        elif self.backbone_str in ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']:
            self.first = self.backbone.conv1 = torch.nn.Conv2d(m,
                                                               64,
                                                               kernel_size=(7, 7),
                                                               stride=(2, 2),
                                                               padding=(3, 3),
                                                               bias=False)
            in_features = self.backbone.fc.in_features
            self.last = self.backbone.fc = torch.nn.Linear(in_features, 1)
        else:
            raise Exception(f'Unknown backbone {self.backbone_str}')

    def forward(self, x):
        out = x
        if self.use_cheaplab:
            out = self.cheaplab(out)
        out = self.backbone(out)
        return out


class AlgaeDataset(torch.utils.data.Dataset):
    def __init__(self, savez):
        npz = np.load(savez)
        self.yes = npz.get('yes')
        self.no = npz.get('no')
        self.yeas = self.yes.shape[-1]
        self.nays = self.no.shape[-1]

    def __len__(self):
        return self.yeas + self.nays

    def __getitem__(self, idx):
        if idx < self.yeas:
            data, label = self.yes[..., idx], 1
        else:
            data, label = self.no[..., idx - self.yeas], 0
        data = data.transpose((2, 0, 1))
        return (data, label)


if __name__ == '__main__':

    args = cli_parser().parse_args()
    logging.basicConfig(stream=sys.stderr, level=logging.INFO)
    log = logging.getLogger()

    device = torch.device('cuda')
    model = AlgaeClassifier(imagery=args.imagery,
                            use_cheaplab=args.cheaplab,
                            backbone_str=args.backbone)
    model.to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    obj = torch.nn.BCEWithLogitsLoss().to(device)

    dl = DataLoader(AlgaeDataset(args.savez), **dataloader_cfg)

    log.info(f'cheaplab={args.cheaplab}')

    if args.pth_load is None:
        log.info('Training everything')
        unfreeze(model.backbone)
        for epoch in range(0, args.epochs1):
            losses = []
            for (i, batch) in enumerate(dl):
                out = model(batch[0].float().to(device)).squeeze()
                constraint = obj(out, batch[1].float().to(device))
                loss = constraint
                losses.append(loss.item())
                loss.backward()
                opt.step()
                opt.zero_grad()
            log.info(f'epoch={epoch} constraint={np.mean(losses)}')

        log.info('Training input filters')
        freeze(model)
        unfreeze(model.first)
        if args.cheaplab:
            unfreeze(model.cheaplab)
        for epoch in range(0, args.epochs1):
            losses = []
            for (i, batch) in enumerate(dl):
                out = model(batch[0].float().to(device)).squeeze()
                constraint = obj(out, batch[1].float().to(device))
                loss = constraint
                losses.append(loss.item())
                loss.backward()
                opt.step()
                opt.zero_grad()
            log.info(f'epoch={epoch} constraint={np.mean(losses)}')

        log.info('Training fully-connected layer')
        freeze(model)
        unfreeze(model.last)
        for epoch in range(0, args.epochs1):
            losses = []
            for (i, batch) in enumerate(dl):
                out = model(batch[0].float().to(device)).squeeze()
                constraint = obj(out, batch[1].float().to(device))
                loss = constraint
                losses.append(loss.item())
                loss.backward()
                opt.step()
                opt.zero_grad()
            log.info(f'epoch={epoch} constraint={np.mean(losses)}')
    else:
        log.info(f'Loading model from {args.pth_load}')
        model.load_state_dict(torch.load(args.pth_load))

    log.info('Training everything')
    unfreeze(model.backbone)
    for epoch in range(0, args.epochs2):
        losses = []
        for (i, batch) in enumerate(dl):
            out = model(batch[0].float().to(device)).squeeze()
            constraint = obj(out, batch[1].float().to(device))
            loss = constraint
            losses.append(loss.item())
            loss.backward()
            opt.step()
            opt.zero_grad()
        log.info(f'epoch={epoch} constraint={np.mean(losses)}')

    if args.pth_save is not None:
        log.info(f'Saving model to {args.pth_save}')
        torch.save(model.state_dict(), args.pth_save)

    model.eval()
    tp = 0.0
    tn = 0.0
    fp = 0.0
    fn = 0.0
    for (i, batch) in enumerate(dl):
        pred = torch.sigmoid(model(batch[0].float().to(device))).squeeze()
        pred = (pred > 0.5).detach().cpu().numpy().astype(np.uint8)
        gt = batch[1].detach().cpu().numpy().astype(np.uint8)
        tp += np.sum((pred == 1) * (gt == 1))
        tn += np.sum((pred == 0) * (gt == 0))
        fp += np.sum((pred == 1) * (gt == 0))
        fn += np.sum((pred == 0) * (gt == 1))
    total = tp + tn + fp + fn
    log.info(f'tpr={tp/(tp+fn)} tnr={tn/(tn+fp)}')
