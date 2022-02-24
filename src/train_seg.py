#!/usr/bin/env python3

import argparse
import logging
import random
import sys

import numpy as np
import torch
import torchvision as tv
import tqdm
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader

from datasets import SegmentationDataset


def cli_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', required=False, type=int, default=4)
    parser.add_argument('--sentinel-l1c-path', required=False, type=str, default=None)
    parser.add_argument('--sentinel-l2a-path', required=False, type=str, default=None)
    parser.add_argument('--aviris-l1-path', required=False, type=str, default=None)
    parser.add_argument('--epochs', required=False, type=int, default=107)
    parser.add_argument('--pseudo-epoch-size', required=False, type=int, default=1007)
    parser.add_argument('--lr', required=False, type=float, default=1e-4)
    parser.add_argument('--num-workers', required=False, type=int, default=1)
    parser.add_argument('--pth-save', required=True, type=str)
    return parser


def worker_init_fn(x):
    np.random.seed(42 + x)
    random.seed(42 + x)


dataloader_cfg = {
    'batch_size': None,
    'num_workers': None,
    'worker_init_fn': worker_init_fn
}


if __name__ == '__main__':

    args = cli_parser().parse_args()
    logging.basicConfig(stream=sys.stderr, level=logging.INFO, format='%(asctime)-15s %(message)s')
    log = logging.getLogger()

    dataloader_cfg['batch_size'] = args.batch_size
    dataloader_cfg['num_workers'] = args.num_workers

    train_dls = []
    valid_dls = []
    if args.sentinel_l1c_path is not None:
        ds = SegmentationDataset(
            args.sentinel_l1c_path,
            is_aviris=False, is_cloud=True, is_validation=False)
        dl = DataLoader(ds, **dataloader_cfg)
        train_dls.append({'dl': dl, 'shadows': False})
        ds = SegmentationDataset(
            args.sentinel_l1c_path,
            is_aviris=False, is_cloud=True, is_validation=True)
        dl = DataLoader(ds, **dataloader_cfg)
        valid_dls.append({'dl': dl, 'shadows': False})
    if args.sentinel_l2a_path is not None:
        ds = SegmentationDataset(
            args.sentinel_l2a_path,
            is_aviris=False, is_cloud=True, is_validation=False)
        dl = DataLoader(ds, **dataloader_cfg)
        train_dls.append({'dl': dl, 'shadows': False})
        ds = SegmentationDataset(
            args.sentinel_l2a_path,
            is_aviris=False, is_cloud=True, is_validation=True)
        dl = DataLoader(ds, **dataloader_cfg)
        valid_dls.append({'dl': dl, 'shadows': False})
    if args.aviris_l1_path is not None:
        ds = SegmentationDataset(
            args.aviris_l1_path,
            is_aviris=True, is_cloud=True, is_validation=False)
        dl = DataLoader(ds, **dataloader_cfg)
        train_dls.append({'dl': dl, 'shadows': True})
        ds = SegmentationDataset(
            args.aviris_l1_path,
            is_aviris=True, is_cloud=True, is_validation=True)
        dl = DataLoader(ds, **dataloader_cfg)
        valid_dls.append({'dl': dl, 'shadows': True})

    from cloud import make_cloud_model
    model = make_cloud_model(in_channels=[12, 13, 224])
    device = torch.device('cuda')
    model.to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    obj_ce = torch.nn.CrossEntropyLoss().to(device)

    batches_per_epoch = args.pseudo_epoch_size // args.batch_size
    for i in range(args.epochs):
        train_losses = []
        model.train()
        for j in tqdm.tqdm(range(batches_per_epoch)):
            choice = random.choice(train_dls)
            dl = iter(choice.get('dl'))
            shadows = choice.get('shadows')
            batch = next(dl)
            out = model(batch[0].to(device))
            label = batch[1]
            if not shadows:
                out = out[:, :2, :, :]
            loss = obj_ce(out, label.to(device))
            train_losses.append(loss.item())
            loss.backward()
        avg_train_loss = np.mean(train_losses)

        valid_losses = []
        model.eval()
        for j in tqdm.tqdm(range(10)):
            choice = random.choice(valid_dls)
            dl = iter(choice.get('dl'))
            shadows = choice.get('shadows')
            batch = next(dl)
            with torch.no_grad():
                out = model(batch[0].to(device))
                label = batch[1]
                if not shadows:
                    out = out[:, :2, :, :]
                loss = obj_ce(out, label.to(device))
                valid_losses.append(loss.item())
        avg_valid_loss = np.mean(valid_losses)

        log.info(f'epoch={i:<3d} avg_train_loss={avg_train_loss:1.5f} avg_valid_loss={avg_valid_loss:1.5f}')

        if i % 33 == 0:
            log.info(f'Saving checkpoint to /tmp/checkpoint.pth')
            torch.save(model.state_dict(), '/tmp/checkpoint.pth')

    if args.pth_save is not None:
        log.info(f'Saving model to {args.pth_save}')
        torch.save(model.state_dict(), args.pth_save)
