#!/usr/bin/env python3

import argparse
import csv
import logging
import math
import sys
import warnings

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
    parser.add_argument('--lr', required=False, type=float, default=1e-4)
    parser.add_argument('--pth-load', required=False, type=str, default=None)
    parser.add_argument('--pth-save', required=False, type=str, default=None)
    parser.add_argument('--savez', required=True, type=str)
    parser.add_argument('--w', required=False, type=float, default=0.0)

    parser.add_argument('--ndwi-mask',
                        required=False,
                        dest='ndwi_mask',
                        action='store_true')
    parser.set_defaults(ndwi_mask=False)

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


def entropy_function(x, w=1e-1):
    s = torch.sigmoid(x)
    pyes_narrow = torch.mean(greens_function(s, 0.25, 1.0 / 16))
    pyes_wide = torch.mean(greens_function(s, 0.25, 1.0 / 8))
    pno_narrow = torch.mean(greens_function(s, 0.75, 1.0 / 16))
    pno_wide = torch.mean(greens_function(s, 0.75, 1.0 / 8))
    return w * ((pyes_narrow * torch.log(pyes_wide)) +
                (pno_narrow * torch.log(pno_wide)))


class AlgaeDataset(torch.utils.data.Dataset):
    def __init__(self, savez, ndwi_mask: bool = False):
        npz = np.load(savez)
        self.yes = npz.get('yes')
        self.no = npz.get('no')
        self.yeas = self.yes.shape[-1]
        self.nays = self.no.shape[-1]
        self.ndwi_mask = ndwi_mask
        warnings.filterwarnings('ignore')

    def __len__(self):
        return self.yeas + self.nays

    def __getitem__(self, idx):
        if idx < self.yeas:
            data, label = self.yes[..., idx], 1
        else:
            data, label = self.no[..., idx - self.yeas], 0
        data = data.transpose((2, 0, 1))
        if self.ndwi_mask:
            ndwi = (data[2] - data[7]) / (data[2] + data[7])
            data *= (ndwi > 0.0)
        return (data, label)


if __name__ == '__main__':

    args = cli_parser().parse_args()
    logging.basicConfig(stream=sys.stderr, level=logging.INFO)
    log = logging.getLogger()

    device = torch.device('cuda')
    model = torch.hub.load('jamesmcclain/algae-classifier:master',
                           'make_algae_model',
                           imagery=args.imagery,
                           use_cheaplab=args.cheaplab,
                           backbone_str=args.backbone)
    model.to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    obj = torch.nn.BCEWithLogitsLoss().to(device)

    dl = DataLoader(AlgaeDataset(args.savez, args.ndwi_mask), **dataloader_cfg)

    log.info(f'backbone={args.backbone}')
    log.info(f'cheaplab={args.cheaplab}')
    log.info(f'epochs1={args.epochs1}')
    log.info(f'epochs2={args.epochs2}')
    log.info(f'imagery={args.imagery}')
    log.info(f'ndwi-mask={args.ndwi_mask}')
    log.info(f'pth-load={args.pth_load}')
    log.info(f'pth-save={args.pth_save}')
    log.info(f'savez={args.savez}')
    log.info(f'parameter lr: {args.lr}')
    log.info(f'parameter w:  {args.w}')

    if args.pth_load is None:
        log.info('Training everything')
        unfreeze(model.backbone)
        for epoch in range(0, args.epochs1):
            losses = []
            for (i, batch) in enumerate(dl):
                out = model(batch[0].float().to(device)).squeeze()
                constraint = obj(out, batch[1].float().to(device))
                entropy = entropy_function(out, args.w)
                loss = constraint + entropy
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
                entropy = entropy_function(out, args.w)
                loss = constraint + entropy
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
                entropy = entropy_function(out, args.w)
                loss = constraint + entropy
                losses.append(loss.item())
                loss.backward()
                opt.step()
                opt.zero_grad()
            log.info(f'epoch={epoch} constraint={np.mean(losses)}')
    else:
        log.info(f'Loading model from {args.pth_load}')
        model.load_state_dict(torch.load(args.pth_load))
        model.to(device)

    log.info('Training everything')
    unfreeze(model.backbone)
    for epoch in range(0, args.epochs2):
        losses = []
        for (i, batch) in enumerate(dl):
            out = model(batch[0].float().to(device)).squeeze()
            constraint = obj(out, batch[1].float().to(device))
            entropy = entropy_function(out, args.w)
            loss = constraint + entropy
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
