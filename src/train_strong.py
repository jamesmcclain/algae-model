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

from datasets import AlgaeSegmentationDataset


BACKBONES = ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']


def cli_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--backbone', required=True, type=str, choices=BACKBONES)
    parser.add_argument('--batch-size', required=False, type=int, default=8)
    parser.add_argument('--train-dataset-path', required=True, type=str)
    parser.add_argument('--valid-dataset-path', required=False, type=str)
    parser.add_argument('--epochs', required=False, type=int, default=107)
    parser.add_argument('--lr', required=False, type=float, default=1e-4)
    parser.add_argument('--num-workers', required=False, type=int, default=0)
    parser.add_argument('--pth-cheaplab-donor', required=False, type=str, default=None)
    parser.add_argument('--pth-save', required=True, type=str)

    parser.add_argument('--freeze-bn', dest='freeze-bn', action='store_true')
    parser.add_argument('--no-freeze-bn', dest='freeze-bn', action='store_false')
    parser.set_defaults(freeze_bn=True)

    parser.add_argument('--freeze-cheaplab', dest='freeze-cheaplab', action='store_true')
    parser.add_argument('--no-freeze-cheaplab', dest='freeze-cheaplab', action='store_false')
    parser.set_defaults(freeze_cheaplab=False)

    return parser


def worker_init_fn(x):
    np.random.seed(42 + x)
    random.seed(42 + x)


dataloader_cfg = {
    'batch_size': None,
    'num_workers': None,
    'shuffle': True,
    'worker_init_fn': worker_init_fn
}


def do_epoch(dl, model, validation_mode: bool = False):
    seg_losses = []
    cls_losses = []
    losses = []

    for (i, batch) in tqdm.tqdm(enumerate(dl)):
        out = model(batch[0].float().to(device))
        seg_out = out.get('seg_out')
        cls_out = out.get('cls_out').get('0')
        loss = 0

        # Segmentation
        seg_label = batch[1].to(device)
        seg_loss = obj_ce(seg_out, seg_label)
        if np.isnan(seg_loss.item()):
            seg_loss = 0
        seg_losses.append(seg_loss.item())
        loss += seg_loss

        # Classification
        cls_label = batch[2].to(device)
        cls_loss = obj_bce(cls_out, cls_label)
        if np.isnan(cls_loss.item()):
            cls_loss = 0
        cls_losses.append(cls_loss.item())
        loss += cls_loss

        losses.append(loss.item())

        if not validation_mode:
            loss.backward()
            opt.step()
            opt.zero_grad()

    return (seg_losses, cls_losses, losses)


if __name__ == '__main__':

    args = cli_parser().parse_args()
    logging.basicConfig(stream=sys.stderr, level=logging.INFO, format='%(asctime)-15s %(message)s')
    log = logging.getLogger()

    dataloader_cfg['batch_size'] = args.batch_size
    dataloader_cfg['num_workers'] = args.num_workers

    device = torch.device('cuda')
    model = torch.hub.load(
        'jamesmcclain/algae-classifier:2a51273a16edaab22645f455e0b91002a74c702b',
        'make_algae_model',
        in_channels=[4, 12, 224],
        backbone=args.backbone,
        pretrained=True)
    if args.pth_cheaplab_donor is not None:
        state = torch.load(args.pth_cheaplab_donor)
        for key in list(state.keys()):
            if 'cheaplab' not in key:
                state.pop(key)
        model.load_state_dict(state, strict=False)
    for cheaplab in model.cheaplab.values():
        cheaplab.to(device)
    model.to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    sched = OneCycleLR(opt, max_lr=args.lr, total_steps=args.epochs)
    obj_ce = torch.nn.CrossEntropyLoss(ignore_index=0xff).to(device)
    obj_bce = torch.nn.BCEWithLogitsLoss().to(device)
    train_dl = DataLoader(AlgaeSegmentationDataset(args.train_dataset_path), **dataloader_cfg)
    if args.valid_dataset_path is not None:
        valid_dl = DataLoader(AlgaeSegmentationDataset(args.valid_dataset_path), **dataloader_cfg)
    else:
        valid_dl = None

    model.whole_mode()
    if args.freeze_cheaplab:
        model.freeze_cheaplab()
    for epoch in range(0, args.epochs):
        if epoch == 1:
            if args.freeze_bn:
                model.freeze_bn()

        model.train()
        seg_losses, cls_losses, losses = do_epoch(train_dl, model, validation_mode=False)
        sched.step()
        log.info(f'Train: epoch={epoch:<3d} loss={np.mean(losses):1.5f} class={np.mean(cls_losses):1.5f} segmentation={np.mean(seg_losses):1.5f}')

        if valid_dl is not None:
            model.eval()
            with torch.no_grad():
                seg_losses, cls_losses, losses = do_epoch(valid_dl, model, validation_mode=True)
            log.info(f' Eval: epoch={epoch:<3d} loss={np.mean(losses):1.5f} class={np.mean(cls_losses):1.5f} segmentation={np.mean(seg_losses):1.5f}')

        if epoch % 33 == 0:
            log.info(f'Saving checkpoint to /tmp/checkpoint.pth')
            torch.save(model.state_dict(), '/tmp/checkpoint.pth')


    if args.pth_save is not None:
        log.info(f'Saving model to {args.pth_save}')
        torch.save(model.state_dict(), args.pth_save)
