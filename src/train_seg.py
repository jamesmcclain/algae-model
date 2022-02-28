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
    parser.add_argument('--batch-size', required=False, type=int, default=8)
    parser.add_argument('--sentinel-l1c-path', required=False, type=str, default=None)
    parser.add_argument('--sentinel-l2a-path', required=False, type=str, default=None)
    parser.add_argument('--aviris-l1-path', required=False, type=str, default=None)
    parser.add_argument('--epochs', required=False, type=int, default=107)
    parser.add_argument('--pseudo-epoch-size', required=False, type=int, nargs='+', default=[10007, 10007, 503])
    parser.add_argument('--lr', required=False, type=float, default=1e-3)
    parser.add_argument('--num-workers', required=False, type=int, default=1)
    parser.add_argument('--pth-save', required=False, type=str, default='model.pth')
    return parser


def worker_init_fn(x):
    np.random.seed(42 + x)
    random.seed(42 + x)


def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)


dataloader_cfg = {
    'batch_size': None,
    'num_workers': None,
    'pin_memory': True,
    'worker_init_fn': worker_init_fn,
    'collate_fn': collate_fn
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

    assert len(train_dls) == len(valid_dls)

    from cloud import make_cloud_model
    model = make_cloud_model(in_channels=[13, 12, 224])
    device = torch.device('cuda')
    model.to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    obj_ce = torch.nn.CrossEntropyLoss().to(device)

    for i in range(args.epochs):
        # choice_index = len(train_dls)-1 if (i < (args.epochs // 5)) else i % len(train_dls)
        choice_index = i % len(train_dls)
        batches_per_epoch = args.pseudo_epoch_size[choice_index] // args.batch_size

        train_losses = []
        choice = train_dls[choice_index]
        dl = choice.get('dl')
        shadows = choice.get('shadows')
        model.train()
        for (j, batch) in tqdm.tqdm(enumerate(dl)):
            out = model(batch[0].to(device))
            if shadows:
                loss = obj_ce(out, batch[1].to(device))
            else:
                loss = obj_ce(out[:, [0, 1], :, :], batch[1].to(device))
            train_losses.append(loss.item())
            loss.backward()
            opt.step()
            opt.zero_grad()
        avg_train_loss = np.mean(train_losses)

        valid_losses = []
        choice = valid_dls[choice_index]
        dl = choice.get('dl')
        shadows = choice.get('shadows')
        model.eval()
        for (j, batch) in tqdm.tqdm(enumerate(dl)):
            with torch.no_grad():
                out = model(batch[0].to(device))
                if shadows:
                    loss = obj_ce(out, batch[1].to(device))
                else:
                    loss = obj_ce(out[:, [0, 1], :, :], batch[1].to(device))
                valid_losses.append(loss.item())
        avg_valid_loss = np.mean(valid_losses)
        log.info(f'epoch={i:<3d} avg_train_loss={avg_train_loss:1.5f} avg_valid_loss={avg_valid_loss:1.5f}')

        if i % 7 == 0:
            log.info(f'Saving checkpoint to /tmp/checkpoint.pth')
            torch.save(model.state_dict(), '/tmp/checkpoint.pth')

    if args.pth_save is not None:
        log.info(f'Saving model to {args.pth_save}')
        torch.save(model.state_dict(), args.pth_save)