#!/usr/bin/env python3

import argparse
import copy
import logging
import sys
import warnings

import numpy as np
import rasterio as rio
import torch
import torch.hub
import tqdm
from rasterio.windows import Window

BACKBONES = [
    'vgg16', 'squeezenet1_0', 'densenet161', 'shufflenet_v2_x1_0',
    'mobilenet_v2', 'mobilenet_v3_large', 'mobilenet_v3_small', 'mnasnet1_0',
    'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152'
]


def cli_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--backbone',
                        required=True,
                        type=str,
                        choices=BACKBONES)
    parser.add_argument('--chunksize', required=False, type=int, default=2048)
    parser.add_argument('--device',
                        required=False,
                        type=str,
                        default='cuda',
                        choices=['cuda', 'cpu'])
    parser.add_argument('--imagery',
                        required=False,
                        type=str,
                        default='sentinel2',
                        choices=['aviris', 'sentinel2'])
    parser.add_argument('--infile', required=True, type=str)
    parser.add_argument('--outfile', required=True, type=str)
    parser.add_argument('--pth-load', required=True, type=str)
    parser.add_argument('--stride', required=False, type=int, default=13)
    parser.add_argument('--window-size', required=False, type=int, default=32)

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

    return parser


if __name__ == '__main__':

    warnings.filterwarnings('ignore')

    args = cli_parser().parse_args()
    logging.basicConfig(stream=sys.stderr, level=logging.INFO)
    log = logging.getLogger()

    device = torch.device(args.device)
    model = torch.hub.load('jamesmcclain/algae-classifier:master',
                           'make_algae_model',
                           imagery=args.imagery,
                           use_cheaplab=args.cheaplab,
                           backbone_str=args.backbone,
                           pretrained=False)
    model.load_state_dict(torch.load(args.pth_load))
    model.to(device)
    model.eval()

    with rio.open(args.infile, 'r') as infile_ds, torch.no_grad():
        out_raw_profile = copy.deepcopy(infile_ds.profile)
        out_raw_profile.update({
            'compress': 'lzw',
            'dtype': np.float32,
            'count': 1,
            'bigtiff': 'yes',
        })
        width = infile_ds.width
        height = infile_ds.height
        ar_out = torch.zeros((1, height, width),
                             dtype=torch.float32).to(device)
        pixel_hits = torch.zeros((1, height, width),
                                 dtype=torch.uint8).to(device)

        if args.imagery == 'aviris':
            indexes = list(range(1, 224 + 1))
        elif args.imagery == 'sentinel2':
            indexes = list(range(1, 12 + 1))
        n = args.window_size

        # gather up batches
        batches = []
        for i in range(0, width - n, args.stride):
            for j in range(0, height - n, args.stride):
                batches.append((i, j))
        batches = [
            batches[i:i + args.chunksize]
            for i in range(0, len(batches), args.chunksize)
        ]

        for batch in tqdm.tqdm(batches):
            windows = [
                infile_ds.read(indexes, window=Window(i, j, n, n))
                for (i, j) in batch
            ]
            windows = [w.astype(np.float32) for w in windows]
            if args.ndwi_mask:
                windows = [
                    w * (((w[2] - w[7]) / (w[2] + w[7])) > 0.0)
                    for w in windows
                ]
            if args.cloud_hack:
                windows = [(w * (w[3] > 100) * (w[3] < 1000)) for w in windows]

            try:
                windows = np.stack(windows, axis=0)
            except:
                continue
            windows = torch.from_numpy(windows).to(dtype=torch.float32,
                                                   device=device)
            prob = torch.sigmoid(model(windows))

            for k, (i, j) in enumerate(batch):
                ar_out[0, j:(j + n), i:(i + n)] += prob[k]
                pixel_hits[0, j:(j + n), i:(i + n)] += 1

    # Bring results back to CPU
    ar_out /= pixel_hits
    ar_out = ar_out.cpu().numpy()

    # Write results to file
    with rio.open(args.outfile, 'w', **out_raw_profile) as outfile_raw_ds:
        outfile_raw_ds.write(ar_out[0],
                             indexes=1,
                             window=Window(0, 0, width, height))
