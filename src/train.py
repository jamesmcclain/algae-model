#!/usr/bin/env python3

import argparse
import csv
import logging
import math
import sys

import numpy as np
import torch
import torch.hub
import torch.nn.functional as F
import torchvision as tv
import tqdm
from torch.utils.data import DataLoader
from torchvision.io import VideoReader
from torchvision.transforms.functional import F_t

from datasets import AlgaeClassificationDataset

BACKBONES = [
    'vgg16', 'densenet161', 'shufflenet_v2_x1_0', 'mobilenet_v2',
    'mobilenet_v3_large', 'mobilenet_v3_small', 'resnet18', 'resnet34',
    'resnet50', 'resnet101', 'resnet152', 'efficientnet_b0', 'efficientnet_b1',
    'efficientnet_b2', 'efficientnet_b3', 'efficientnet_b4', 'efficientnet_b5',
    'efficientnet_b6', 'efficientnet_b7'
]


def cli_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--backbone', required=True, type=str, choices=BACKBONES)
    parser.add_argument('--classification-batch-size', required=False, type=int, default=128)
    parser.add_argument('--unlabeled-batch-size', required=False, type=int, default=4096)
    parser.add_argument('--classification-savezs', required=True, type=str, nargs='+')
    parser.add_argument('--epochs1', required=False, type=int, default=33)
    parser.add_argument('--epochs2', required=False, type=int, default=2003)
    parser.add_argument('--lr1', required=False, type=float, default=1e-4)
    parser.add_argument('--lr2', required=False, type=float, default=1e-5)
    parser.add_argument('--num-workers', required=False, type=int, default=0)
    parser.add_argument('--prescale', required=False, type=int, default=1)
    parser.add_argument('--pth-cheaplab-donor', required=False, type=str, default=None)
    parser.add_argument('--pth-load', required=False, type=str, default=None)
    parser.add_argument('--pth-save', required=False, type=str, default=None)
    parser.add_argument('--w0', required=False, type=float, default=1.0)
    parser.add_argument('--w1', required=False, type=float, default=0.0)

    parser.add_argument('--ndwi-mask',
                        required=False,
                        dest='ndwi_mask',
                        action='store_true')
    parser.set_defaults(ndwi_mask=False)

    parser.add_argument('--no-cloud-hack',
                        dest='cloud_hack',
                        action='store_false')
    parser.set_defaults(cloud_hack=True)

    parser.add_argument('--no-schedule', dest='schedule', action='store_false')
    parser.set_defaults(schedule=True)

    parser.add_argument('--no-pretrained',
                        dest='pretrained',
                        action='store_false')
    parser.set_defaults(pretrained=True)

    return parser


def worker_init_fn(x):
    np.random.seed(42 + x)


dataloader1_cfg = {
    'batch_size': None,
    'num_workers': None,
    'shuffle': True,
    'worker_init_fn': worker_init_fn
}

dataloader2_cfg = {
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


if __name__ == '__main__':

    args = cli_parser().parse_args()
    logging.basicConfig(stream=sys.stderr, level=logging.INFO)
    log = logging.getLogger()

    dataloader1_cfg['batch_size'] = args.classification_batch_size
    dataloader1_cfg['num_workers'] = args.num_workers

    dataloader2_cfg['batch_size'] = args.unlabeled_batch_size
    dataloader2_cfg['num_workers'] = args.num_workers

    device = torch.device('cuda')
    model = torch.hub.load(
        'jamesmcclain/algae-classifier:e8f671fa063e9d7575040fe93fde306210f160e7',
        'make_algae_model',
        in_channels=[4, 12, 224],
        prescale=args.prescale,
        backbone_str=args.backbone,
        pretrained=args.pretrained)

    if args.pth_cheaplab_donor:
        state = torch.load(args.pth_cheaplab_donor)
        for key in list(state.keys()):
            if 'cheaplab' not in key:
                state.pop(key)
        model.load_state_dict(state, strict=False)

    for cheaplab in model.cheaplab.values():
        cheaplab.to(device)
    model.to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr1)
    obj = torch.nn.BCEWithLogitsLoss().to(device)

    ad = AlgaeClassificationDataset(savezs=args.classification_savezs,
                                    ndwi_mask=args.ndwi_mask,
                                    cloud_hack=args.cloud_hack,
                                    augment=True)
    dl1 = DataLoader(ad, **dataloader1_cfg)

    log.info(f'backbone={args.backbone}')
    log.info(f'classification-batch-size={args.classification_batch_size}')
    log.info(f'classification-savezs={args.classification_savezs}')
    log.info(f'cloud-hack={args.cloud_hack}')
    log.info(f'epochs1={args.epochs1}')
    log.info(f'epochs2={args.epochs2}')
    log.info(f'ndwi-mask={args.ndwi_mask}')
    log.info(f'num-workers={args.num_workers}')
    log.info(f'prescale={args.prescale}')
    log.info(f'pretrained={args.pretrained}')
    log.info(f'pth-load={args.pth_load}')
    log.info(f'pth-save={args.pth_save}')
    log.info(f'schedule={args.schedule}')

    log.info(f'parameter lr1: {args.lr1}')
    log.info(f'parameter lr2: {args.lr2}')
    log.info(f'parameter w0:   {args.w0}')
    log.info(f'parameter w1:   {args.w1}')

    if args.pth_load is None:
        log.info('Training everything')
        unfreeze(model)
        for epoch in range(0, args.epochs1):
            losses = []
            constraints = []
            entropies = []
            for (i, batch) in enumerate(dl1):
                out = model(batch[0].float().to(device)).squeeze()
                constraint = obj(out, batch[1].float().to(device))
                entropy = entropy_function(out)
                loss = args.w0 * constraint + args.w1 * entropy
                losses.append(loss.item())
                entropies.append(entropy.item())
                constraints.append(constraint.item())
                loss.backward()
                opt.step()
                opt.zero_grad()
            mean_loss = np.mean(losses)
            mean_constraint = np.mean(constraints)
            mean_entropy = np.mean(entropies)
            log.info(f'epoch={epoch} loss={mean_loss} entropy={mean_entropy} constraint={mean_constraint}')

        log.info('Training input filters and fully-connected layer')
        freeze(model)
        unfreeze(model.first)
        unfreeze(model.last)
        for epoch in range(0, args.epochs1):
            losses = []
            constraints = []
            entropies = []
            for (i, batch) in enumerate(dl1):
                out = model(batch[0].float().to(device)).squeeze()
                constraint = obj(out, batch[1].float().to(device))
                entropy = entropy_function(out)
                loss = constraint
                losses.append(loss.item())
                entropies.append(entropy.item())
                constraints.append(constraint.item())
                loss.backward()
                opt.step()
                opt.zero_grad()
            mean_loss = np.mean(losses)
            mean_constraint = np.mean(constraints)
            mean_entropy = np.mean(entropies)
            log.info(f'epoch={epoch} loss={mean_loss} entropy={mean_entropy} constraint={mean_constraint}')
    else:
        log.info(f'Loading model from {args.pth_load}')
        model.load_state_dict(torch.load(args.pth_load))
        for cheaplab in model.cheaplab.values():
            cheaplab.to(device)
        model.to(device)

    log.info('Training everything')
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr2)
    if args.epochs2 > 0:
        sched = torch.optim.lr_scheduler.OneCycleLR(opt,
                                                    max_lr=args.lr2,
                                                    total_steps=args.epochs2)
    unfreeze(model.backbone)
    for epoch in range(0, args.epochs2):
        losses = []
        constraints = []
        entropies = []
        for (i, batch) in enumerate(dl1):
            out = model(batch[0].float().to(device)).squeeze()
            constraint = obj(out, batch[1].float().to(device))
            entropy = entropy_function(out)
            loss = args.w0 * constraint + args.w1 * entropy
            losses.append(loss.item())
            entropies.append(entropy.item())
            constraints.append(constraint.item())
            loss.backward()
            opt.step()
            opt.zero_grad()
        if args.schedule:
            sched.step()
        if epoch % 107 == 0:
            log.info(f'Saving checkpoint to /tmp/checkpoint.pth')
            torch.save(model.state_dict(), '/tmp/checkpoint.pth')
        mean_loss = np.mean(losses)
        mean_constraint = np.mean(constraints)
        mean_entropy = np.mean(entropies)
        log.info(f'epoch={epoch} loss={mean_loss} entropy={mean_entropy} constraint={mean_constraint}')

    if args.pth_save is not None:
        log.info(f'Saving model to {args.pth_save}')
        torch.save(model.state_dict(), args.pth_save)

    model.eval()
    tp = 0.0
    tn = 0.0
    fp = 0.0
    fn = 0.0
    with torch.no_grad():
        for (i, batch) in enumerate(dl1):
            pred = torch.sigmoid(model(batch[0].float().to(device))).squeeze()
            pred = (pred > 0.5).detach().cpu().numpy().astype(np.uint8)
            gt = batch[1].detach().cpu().numpy().astype(np.uint8)
            tp += np.sum((pred == 1) * (gt == 1))
            tn += np.sum((pred == 0) * (gt == 0))
            fp += np.sum((pred == 1) * (gt == 0))
            fn += np.sum((pred == 0) * (gt == 1))
    total = tp + tn + fp + fn
    log.info(f'tpr={tp/(tp+fn)} tnr={tn/(tn+fp)}')
