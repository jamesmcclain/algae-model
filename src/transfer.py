#!/usr/bin/env python3

import argparse
import copy
import csv
import logging
import math
import sys
import warnings

import numpy as np
import rasterio as rio
import rasterio.windows
import torch
import torch.hub
import torch.nn.functional as F
import torchvision as tv
import tqdm
from torch.utils.data import DataLoader
from torchvision.io import VideoReader
from torchvision.transforms.functional import F_t

BACKBONES = [
    'vgg16', 'densenet161', 'shufflenet_v2_x1_0', 'mobilenet_v2',
    'mobilenet_v3_large', 'mobilenet_v3_small', 'resnet18', 'resnet34',
    'resnet50', 'resnet101', 'resnet152', 'efficientnet_b0', 'efficientnet_b1',
    'efficientnet_b2', 'efficientnet_b3', 'efficientnet_b4', 'efficientnet_b5',
    'efficientnet_b6', 'efficientnet_b7'
]


def cli_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--backbone',
                        required=True,
                        type=str,
                        choices=BACKBONES)
    parser.add_argument('--imagery',
                        required=False,
                        type=str,
                        default='aviris',
                        choices=['aviris', 'planet'])
    parser.add_argument('--batch-size', required=False, type=int, default=4096)
    parser.add_argument('--epochs', required=False, type=int, default=2003)
    parser.add_argument('--lr', required=False, type=float, default=1e-4)
    parser.add_argument('--num-workers', required=False, type=int, default=0)
    parser.add_argument('--prescale', required=False, type=int, default=1)
    parser.add_argument('--pth-load', required=True, type=str)
    parser.add_argument('--pth-save', required=True, type=str)
    parser.add_argument('--savez', required=True, type=str)

    parser.add_argument('--no-augment', required=False, dest='augment', action='store_false')
    parser.set_defaults(augment=True)

    parser.add_argument('--no-keep-cheaplab', required=False, dest='keep_cheaplab', action='store_false')
    parser.set_defaults(keep_cheaplab=True)

    parser.add_argument('--ndwi-mask', required=False, dest='ndwi_mask', action='store_true')
    parser.set_defaults(ndwi_mask=False)

    parser.add_argument('--cloud-hack', dest='cloud_hack', action='store_true')
    parser.set_defaults(cloud_hack=False)

    parser.add_argument('--schedule', dest='schedule', action='store_true')
    parser.set_defaults(schedule=False)

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


def entropy_function(x):
    s = torch.sigmoid(x)
    pno_narrow = torch.mean(greens_function(s, 0.25, 1.0 / 16))
    pno_wide = torch.mean(greens_function(s, 0.25, 1.0 / 8))
    pyes_narrow = torch.mean(greens_function(s, 0.75, 1.0 / 16))
    pyes_wide = torch.mean(greens_function(s, 0.75, 1.0 / 8))
    return ((pyes_narrow * torch.log(pyes_wide)) +
            (pno_narrow * torch.log(pno_wide)))


class AlgaeDataset(torch.utils.data.Dataset):
    def __init__(self,
                 savez,
                 ndwi_mask: bool = False,
                 cloud_hack: bool = False,
                 augment: bool = False,
                 imagery: str = 'aviris'):
        npz = np.load(savez)
        self.yesno = npz.get('yesno')
        self.yesnos = len(self.yesno)
        self.ndwi_mask = ndwi_mask
        self.cloud_hack = cloud_hack
        self.augment = augment
        self.imagery = imagery
        self.band_count = 224 if self.imagery == 'aviris' else 4
        warnings.filterwarnings('ignore')

    def __len__(self):
        return self.yesnos

    def __getitem__(self, idx):
        data = self.yesno[..., idx]
        data = data.transpose((2, 0, 1))

        # Sentinel-2 water mask
        if self.ndwi_mask:
            if self.imagery == 'aviris':
                ndwi = (data[22] - data[50]) / (data[22] + data[50])
                data *= (ndwi > 0.0)
            elif self.imagery == 'planet':
                ndwi = (data[1] - data[3]) / (data[1] + data[3])
                data *= (ndwi > 0.0)
            else:
                raise Exception(self.imagery)

        # Sentinel-2 cloud mask
        if self.cloud_hack:
            if self.imagery == 'aviris':
                data *= ((data[33] > 600) * (data[33] < 2000))
            elif self.imagery == 'planet':
                data *= ((data[2] > 900) * (data[2] < 4000))
            else:
                raise Exception(self.imagery)

        # Augmentations
        if self.augment:
            if np.random.randint(0, 2) == 0:
                data = np.transpose(data, axes=(0, 2, 1))
            if np.random.randint(0, 2) == 0:
                data = np.flip(data, axis=(1 + np.random.randint(0, 2)))
            if np.random.randint(0, 2) == 0:
                data = np.transpose(data, (0, 2, 1))
            if np.random.randint(0, 5) < 1:
                data *= (1.0 + ((np.random.rand(self.band_count, 1, 1) - 0.5) / 50))
            if np.random.randint(0, 5) < 1:
                data *= (1.0 + ((np.random.rand(self.band_count, 32, 32) - 0.5) / 500))
            data = np.rot90(data, k=np.random.randint(0, 4), axes=(1, 2))
            if np.random.randint(0, 2) == 0:
                data = np.transpose(data, (0, 2, 1))
            data = data.copy()

        return data


if __name__ == '__main__':

    args = cli_parser().parse_args()
    logging.basicConfig(stream=sys.stderr, level=logging.INFO)
    log = logging.getLogger()

    dataloader_cfg['batch_size'] = args.batch_size
    dataloader_cfg['num_workers'] = args.num_workers

    device = torch.device('cuda')
    model = torch.hub.load(
        'jamesmcclain/algae-classifier:b1afcfe5ea32f937ccdbde5751a57c1dbe17ec13',
        'make_algae_model',
        imagery='sentinel2',
        prescale=args.prescale,
        use_cheaplab=True,
        backbone_str=args.backbone,
        pretrained=False)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    obj = torch.nn.BCEWithLogitsLoss().to(device)

    ad = AlgaeDataset(savez=args.savez,
                      ndwi_mask=args.ndwi_mask,
                      cloud_hack=args.cloud_hack,
                      augment=args.augment,
                      imagery=args.imagery)
    dl = DataLoader(ad, **dataloader_cfg)

    log.info(f'augment={args.augment}')
    log.info(f'backbone={args.backbone}')
    log.info(f'batch-size={args.batch_size}')
    log.info(f'cloud-hack={args.cloud_hack}')
    log.info(f'epochs={args.epochs}')
    log.info(f'keep-cheaplab={args.keep_cheaplab}')
    log.info(f'ndwi-mask={args.ndwi_mask}')
    log.info(f'num-workers={args.num_workers}')
    log.info(f'prescale={args.prescale}')
    log.info(f'pth-load={args.pth_load}')
    log.info(f'pth-save={args.pth_save}')
    log.info(f'savez={args.savez}')
    log.info(f'schedule={args.schedule}')

    log.info(f'parameter lr1: {args.lr}')

    log.info(f'Loading model from {args.pth_load}')
    model.load_state_dict(torch.load(args.pth_load))

    if args.keep_cheaplab:
        # Modify CheapLab "learned indices" portion
        sentinel_tensor = model.cheaplab.indices.conv1.weight.data.cpu().numpy()
        (intermediate, channels, _, _) = sentinel_tensor.shape
        assert (channels == 12)
        if args.imagery == 'aviris':
            aviris_tensor = np.zeros((intermediate, 224, 1, 1), dtype=np.float32)
            for i in range(0, intermediate):
                aviris_tensor[i, 10, 0, 0] = sentinel_tensor[i, 0, 0, 0]
                aviris_tensor[i, 15, 0, 0] = sentinel_tensor[i, 1, 0, 0]
                aviris_tensor[i, 22, 0, 0] = sentinel_tensor[i, 2, 0, 0]
                aviris_tensor[i, 33, 0, 0] = sentinel_tensor[i, 3, 0, 0]
                aviris_tensor[i, 37, 0, 0] = sentinel_tensor[i, 4, 0, 0]
                aviris_tensor[i, 41, 0, 0] = sentinel_tensor[i, 5, 0, 0]
                aviris_tensor[i, 45, 0, 0] = sentinel_tensor[i, 6, 0, 0]
                aviris_tensor[i, 50, 0, 0] = sentinel_tensor[i, 7, 0, 0]
                aviris_tensor[i, 62, 0, 0] = sentinel_tensor[i, 8, 0, 0]
                aviris_tensor[i, 107, 0, 0] = sentinel_tensor[i, 9, 0, 0]
                aviris_tensor[i, 132, 0, 0] = sentinel_tensor[i, 10, 0, 0]
                aviris_tensor[i, 193, 0, 0] = sentinel_tensor[i, 11, 0, 0]
            model.cheaplab.indices.conv1 = torch.nn.Conv2d(224,
                                                           intermediate,
                                                           kernel_size=1,
                                                           padding=0,
                                                           bias=False)
            model.cheaplab.indices.conv1.weight.data = torch.tensor(aviris_tensor)
            model.cheaplab.indices.conv1 = model.cheaplab.indices.conv1.to(device)
        elif args.imagery == 'planet':
            planet_tensor = np.zeros((intermediate, 4, 1, 1), dtype=np.float32)
            for i in range(0, intermediate):
                planet_tensor[i, 0, 0, 0] = sentinel_tensor[i, 1, 0, 0]
                planet_tensor[i, 1, 0, 0] = sentinel_tensor[i, 2, 0, 0]
                planet_tensor[i, 2, 0, 0] = sentinel_tensor[i, 3, 0, 0]
                planet_tensor[i, 3, 0, 0] = sentinel_tensor[i, 8, 0, 0]
            model.cheaplab.indices.conv1 = torch.nn.Conv2d(4,
                                                           intermediate,
                                                           kernel_size=1,
                                                           padding=0,
                                                           bias=False)
            model.cheaplab.indices.conv1.weight.data = torch.tensor(planet_tensor)
            model.cheaplab.indices.conv1 = model.cheaplab.indices.conv1.to(device)
        else:
            raise Exception(args.imagery)

        # Modify CheapLab "classifier" portion
        sentinel_tensor = model.cheaplab.classifier[0].conv2d.weight.data.cpu().numpy()
        (intermediate, channels, _, _) = sentinel_tensor.shape
        assert (channels == 44)
        if args.imagery == 'aviris':
            aviris_tensor = np.zeros((intermediate, 32 + 224, 1, 1), dtype=np.float32)
            for i in range(0, intermediate):
                aviris_tensor[i, 32 + 10, 0, 0] = sentinel_tensor[i, 32 + 0, 0, 0]
                aviris_tensor[i, 32 + 15, 0, 0] = sentinel_tensor[i, 32 + 1, 0, 0]
                aviris_tensor[i, 32 + 22, 0, 0] = sentinel_tensor[i, 32 + 2, 0, 0]
                aviris_tensor[i, 32 + 33, 0, 0] = sentinel_tensor[i, 32 + 3, 0, 0]
                aviris_tensor[i, 32 + 37, 0, 0] = sentinel_tensor[i, 32 + 4, 0, 0]
                aviris_tensor[i, 32 + 41, 0, 0] = sentinel_tensor[i, 32 + 5, 0, 0]
                aviris_tensor[i, 32 + 45, 0, 0] = sentinel_tensor[i, 32 + 6, 0, 0]
                aviris_tensor[i, 32 + 50, 0, 0] = sentinel_tensor[i, 32 + 7, 0, 0]
                aviris_tensor[i, 32 + 62, 0, 0] = sentinel_tensor[i, 32 + 8, 0, 0]
                aviris_tensor[i, 32 + 107, 0, 0] = sentinel_tensor[i, 32 + 9, 0, 0]
                aviris_tensor[i, 32 + 132, 0, 0] = sentinel_tensor[i, 32 + 10, 0, 0]
                aviris_tensor[i, 32 + 193, 0, 0] = sentinel_tensor[i, 32 + 11, 0, 0]
            model.cheaplab.classifier[0].conv2d = torch.nn.Conv2d(32 + 224,
                                                                  intermediate,
                                                                  kernel_size=1)
            model.cheaplab.classifier[0].conv2d.weight.data = torch.tensor(aviris_tensor)
            model.cheaplab.classifier[0].conv2d = model.cheaplab.classifier[0].conv2d.to(device)
        elif args.imagery == 'planet':
            planet_tensor = np.zeros((intermediate, 32 + 4, 1, 1), dtype=np.float32)
            for i in range(0, intermediate):
                planet_tensor[i, 32 + 0, 0, 0] = sentinel_tensor[i, 32 + 1, 0, 0]
                planet_tensor[i, 32 + 1, 0, 0] = sentinel_tensor[i, 32 + 2, 0, 0]
                planet_tensor[i, 32 + 2, 0, 0] = sentinel_tensor[i, 32 + 3, 0, 0]
                planet_tensor[i, 32 + 3, 0, 0] = sentinel_tensor[i, 32 + 8, 0, 0]
            model.cheaplab.classifier[0].conv2d = torch.nn.Conv2d(32 + 4,
                                                                  intermediate,
                                                                  kernel_size=1)
            model.cheaplab.classifier[0].conv2d.weight.data = torch.tensor(planet_tensor)
            model.cheaplab.classifier[0].conv2d = model.cheaplab.classifier[0].conv2d.to(device)
        else:
            raise Exception(args.imagery)
    else:
        if args.imagery == 'aviris':
            model.cheaplab = torch.hub.load('jamesmcclain/CheapLab:master',
                                            'make_cheaplab_model',
                                            num_channels=224,
                                            out_channels=3)
        elif args.imagery == 'planet':
            model.cheaplab = torch.hub.load('jamesmcclain/CheapLab:master',
                                            'make_cheaplab_model',
                                            num_channels=4,
                                            out_channels=3)
        else:
            raise Exception(args.imagery)

    model.to(device)

    log.info('Training CheapLab')
    if args.schedule:
        sched = torch.optim.lr_scheduler.OneCycleLR(opt,
                                                    max_lr=args.lr,
                                                    total_steps=args.epochs)

    freeze(model.backbone)
    unfreeze(model.cheaplab)
    for epoch in range(0, args.epochs):
        losses = []
        entropies = []
        for (i, batch) in enumerate(dl):
            out = model(batch.float().to(device)).squeeze()
            entropy = entropy_function(out)
            loss = entropy
            losses.append(loss.item())
            entropies.append(entropy.item())
            loss.backward()
            opt.step()
            opt.zero_grad()
        if args.schedule:
            sched.step()
        if epoch % 107 == 0:
            log.info(f'Saving checkpoint to /tmp/checkpoint_{epoch}.pth')
            torch.save(model.state_dict(), f'/tmp/checkpoint_{epoch}.pth')
        mean_loss = np.mean(losses)
        mean_entropy = np.mean(entropies)
        log.info(f'epoch={epoch} loss={mean_loss} entropy={mean_entropy}')

    if args.pth_save is not None:
        log.info(f'Saving model to {args.pth_save}')
        torch.save(model.state_dict(), args.pth_save)
