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

BACKBONES = [
    'vgg16', 'squeezenet1_0', 'densenet161', 'shufflenet_v2_x1_0',
    'mobilenet_v2', 'mobilenet_v3_large', 'mobilenet_v3_small', 'mnasnet1_0',
    'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152'
]


def cli_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--backbone',
                        required=False,
                        type=str,
                        choices=BACKBONES,
                        default='resnet18')
    parser.add_argument('--batch-size', required=False, type=int, default=128)
    parser.add_argument('--epochs1', required=False, type=int, default=33)
    parser.add_argument('--epochs2', required=False, type=int, default=72)
    parser.add_argument('--imagery',
                        required=False,
                        choices=['aviris', 'sentinel2'],
                        default='aviris')
    parser.add_argument('--lr1', required=False, type=float, default=1e-4)
    parser.add_argument('--lr2', required=False, type=float, default=1e-4)
    parser.add_argument('--num-workers', required=False, type=int, default=8)
    parser.add_argument('--pth-load', required=False, type=str, default=None)
    parser.add_argument('--pth-save', required=False, type=str, default=None)
    parser.add_argument('--savez', required=True, type=str)
    parser.add_argument('--w0', required=False, type=float, default=1.0)
    parser.add_argument('--w', required=False, type=float, default=0.0)

    parser.add_argument('--ndwi-mask',
                        required=False,
                        dest='ndwi_mask',
                        action='store_true')
    parser.set_defaults(ndwi_mask=False)

    parser.add_argument('--cloud-hack',
                        required=False,
                        dest='cloud_hack',
                        action='store_true')
    parser.set_defaults(cloud_hack=False)

    parser.add_argument('--no-cheaplab', dest='cheaplab', action='store_false')
    parser.set_defaults(cheaplab=True)

    parser.add_argument('--no-pretrained',
                        dest='pretrained',
                        action='store_false')
    parser.set_defaults(pretrained=True)

    return parser


def worker_init_fn(x):
    np.random.seed(42 + x)


dataloader_cfg = {
    'batch_size': None,
    'num_workers': None,
    'shuffle': True,
    'worker_init_fn': worker_init_fn
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
    def __init__(self,
                 savez,
                 ndwi_mask: bool = False,
                 cloud_hack: bool = False):
        npz = np.load(savez)
        self.yes = npz.get('yes')
        self.no = npz.get('no')
        self.yeas = self.yes.shape[-1]
        self.nays = self.no.shape[-1]
        self.ndwi_mask = ndwi_mask
        self.cloud_hack = cloud_hack
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

        if self.cloud_hack:
            data *= ((data[3] > 100) * (data[3] < 1000))

        # Augmentations
        rn = np.random.randint(0, 37)
        if (rn % 2) < 1:
            data = np.transpose(data, axes=(0, 2, 1))
        if (rn % 3) < 2:
            data = np.flip(data, axis=(1 + (rn % 2)))
        if (rn % 5) < 3:
            data = np.transpose(data, (0, 2, 1))
        if (rn % 13) == 0:
            data *= 1.033
        elif (rn % 13) == 1:
            data /= 1.072
        data = np.rot90(data, k=(rn % 4), axes=(1, 2)).copy()

        return (data, label)


if __name__ == '__main__':

    args = cli_parser().parse_args()
    logging.basicConfig(stream=sys.stderr, level=logging.INFO)
    log = logging.getLogger()

    dataloader_cfg['batch_size'] = args.batch_size
    dataloader_cfg['num_workers'] = args.num_workers

    device = torch.device('cuda')
    model = torch.hub.load('jamesmcclain/algae-classifier:master',
                           'make_algae_model',
                           imagery=args.imagery,
                           use_cheaplab=args.cheaplab,
                           backbone_str=args.backbone,
                           pretrained=args.pretrained)
    model.to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr1)
    obj = torch.nn.BCEWithLogitsLoss().to(device)

    dl = DataLoader(
        AlgaeDataset(savez=args.savez,
                     ndwi_mask=args.ndwi_mask,
                     cloud_hack=args.cloud_hack), **dataloader_cfg)

    log.info(f'backbone={args.backbone}')
    log.info(f'batch-size={args.batch_size}')
    log.info(f'cheaplab={args.cheaplab}')
    log.info(f'cloud-hack={args.cloud_hack}')
    log.info(f'epochs1={args.epochs1}')
    log.info(f'epochs2={args.epochs2}')
    log.info(f'imagery={args.imagery}')
    log.info(f'ndwi-mask={args.ndwi_mask}')
    log.info(f'num-workers={args.num_workers}')
    log.info(f'pretrained={args.pretrained}')
    log.info(f'pth-load={args.pth_load}')
    log.info(f'pth-save={args.pth_save}')
    log.info(f'savez={args.savez}')

    log.info(f'parameter lr1: {args.lr1}')
    log.info(f'parameter lr2: {args.lr2}')
    log.info(f'parameter w0:  {args.w0}')
    log.info(f'parameter w:   {args.w}')

    if args.pth_load is None:
        log.info('Training everything')
        unfreeze(model.backbone)
        for epoch in range(0, args.epochs1):
            losses = []
            for (i, batch) in enumerate(dl):
                out = model(batch[0].float().to(device)).squeeze()
                constraint = obj(out, batch[1].float().to(device))
                entropy = entropy_function(out, args.w)
                loss = args.w0 * constraint + entropy
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
                loss = args.w0 * constraint + entropy
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
                loss = args.w0 * constraint + entropy
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
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr2)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt,
                                                max_lr=args.lr2,
                                                total_steps=args.epochs2)
    unfreeze(model.backbone)
    for epoch in range(0, args.epochs2):
        losses = []
        for (i, batch) in enumerate(dl):
            out = model(batch[0].float().to(device)).squeeze()
            constraint = obj(out, batch[1].float().to(device))
            entropy = entropy_function(out, args.w)
            loss = args.w0 * constraint + entropy
            losses.append(loss.item())
            loss.backward()
            opt.step()
            opt.zero_grad()
        sched.step()
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
