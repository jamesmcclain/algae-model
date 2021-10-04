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


def cli_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--chunksize', required=False, type=int, default=2048)
    parser.add_argument('--device', required=False, type=str, default='cuda', choices=['cuda', 'cpu'])
    parser.add_argument('--infile', required=True, type=str, nargs='+')
    parser.add_argument('--outfile', required=True, type=str, nargs='+')
    parser.add_argument('--prescale', required=False, type=int, default=1)
    parser.add_argument('--pth-load', required=True, type=str)
    parser.add_argument('--window-size', required=False, type=int, default=32)

    parser.add_argument('--ndwi-mask', required=False, dest='ndwi_mask', action='store_true')
    parser.set_defaults(ndwi_mask=False)

    parser.add_argument('--no-cloud-hack', dest='cloud_hack', action='store_false')
    parser.set_defaults(cloud_hack=True)

    return parser


if __name__ == '__main__':

    warnings.filterwarnings('ignore')

    args = cli_parser().parse_args()
    logging.basicConfig(stream=sys.stderr, level=logging.INFO)
    log = logging.getLogger()

    n = args.window_size

    device = torch.device(args.device)
    model = torch.hub.load('jamesmcclain/algae-classifier:ee445e3bf5beee56a47d2d23a6c69aa627dc3679',
                           'make_algae_model',
                           in_channels=[4, 12, 224],
                           prescale=args.prescale,
                           backbone_str='resnet18',
                           pretrained=False)
    state = torch.load(args.pth_load)
    for key in list(state.keys()):
        if 'cheaplab' not in key:
            state.pop(key)
    model.load_state_dict(state, strict=False)
    model = model.cheaplab
    model.to(device)
    model.eval()

    # Read data
    for (infile, outfile) in zip(args.infile, args.outfile):
        with rio.open(infile, 'r') as infile_ds, torch.no_grad():
            out_raw_profile = copy.deepcopy(infile_ds.profile)
            out_raw_profile.update({
                'compress': 'lzw',
                'dtype': np.float32,
                'count': 3,
                'bigtiff': 'yes',
                'sparse_ok': 'yes',
                'tiled': 'yes'
            })
            width = infile_ds.width
            height = infile_ds.height
            bandcount = infile_ds.count

            data_out = torch.zeros((3, height, width),
                                dtype=torch.float32).to(device)

            if bandcount == 224:
                indexes = list(range(1, 224 + 1))
            elif bandcount in {12, 13}:
                indexes = list(range(1, 12 + 1))
                # XXX 13 bands does not indicate L1C support, this is
                # for Franklin COGs that have an extra band.
                bandcount = 12
            elif bandcount == 4:
                indexes = list(range(1, 4 + 1))
            elif bandcount == 5:
                indexes = [1, 2, 3, 5]
                bandcount = 4
            else:
                raise Exception(f'bands={bandcount}')

            # Gather up batches
            batches = []
            for i in range(0, width - n, n):
                for j in range(0, height - n, n):
                    batches.append((i, j))
            batches = [
                batches[i:i + args.chunksize]
                for i in range(0, len(batches), args.chunksize)
            ]

            # Evaluate batches
            for batch in tqdm.tqdm(batches):
                windows = [
                    infile_ds.read(indexes, window=Window(i, j, n, n))
                    for (i, j) in batch
                ]
                windows = [w.astype(np.float32) for w in windows]
                if bandcount == 12:
                    if args.ndwi_mask:
                        windows = [
                            w * (((w[2] - w[7]) / (w[2] + w[7])) > 0.0)
                            for w in windows
                        ]
                    if args.cloud_hack:
                        windows = [(w * (w[3] > 100) * (w[3] < 1000)) for w in windows]
                elif bandcount == 224:
                    if args.ndwi_mask:
                        windows = [
                            w * (((w[22] - w[50]) / (w[22] + w[50])) > 0.0)
                            for w in windows
                        ]
                    if args.cloud_hack:
                        windows = [(w * (w[33] > 600) * (w[33] < 2000)) for w in windows]
                elif bandcount == 4:
                    if args.ndwi_mask:
                        windows = [
                            w * (((w[1] - w[3]) / (w[1] + w[3])) > 0.0)
                            for w in windows
                        ]
                    if args.cloud_hack:
                        windows = [(w * (w[2] > 900) * (w[2] < 4000)) for w in windows]

                try:
                    windows = np.stack(windows, axis=0)
                except:
                    continue
                if windows.sum() == 0:
                    continue
                windows = torch.from_numpy(windows).to(dtype=torch.float32, device=device)
                prob = torch.sigmoid(model[str(bandcount)](windows))

                for k, (i, j) in enumerate(batch):
                    data_out[:, j:(j + n), i:(i + n)] = prob[k]

            # Bring results back to CPU
            data_out = data_out.cpu().numpy()

        # Write results to file
        with rio.open(outfile, 'w', **out_raw_profile) as outfile_raw_ds:
            outfile_raw_ds.write(data_out)
