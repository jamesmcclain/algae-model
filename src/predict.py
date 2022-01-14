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


BACKBONES = ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']


def cli_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--backbone', required=True, type=str, choices=BACKBONES)
    parser.add_argument('--chunksize', required=False, type=int, default=4)
    parser.add_argument('--device', required=False, type=str, default='cuda', choices=['cuda', 'cpu'])
    parser.add_argument('--infile', required=True, type=str, nargs='+')
    parser.add_argument('--outfile', required=False, default=None, type=str, nargs='+')
    parser.add_argument('--pth-load', required=True, type=str)
    parser.add_argument('--stride', required=False, type=int, default=257)
    parser.add_argument('--window-size', required=False, type=int, default=512)

    parser.add_argument('--ndwi-mask', required=False, dest='ndwi_mask', action='store_true')
    parser.set_defaults(ndwi_mask=False)

    parser.add_argument('--force-class', required=False, dest='force_class', action='store_true')
    parser.set_defaults(force_class=False)

    return parser


if __name__ == '__main__':

    warnings.filterwarnings('ignore')

    args = cli_parser().parse_args()
    logging.basicConfig(stream=sys.stderr, level=logging.INFO, format='%(asctime)-15s %(message)s')
    log = logging.getLogger()

    n = args.window_size

    device = torch.device(args.device)
    model = torch.hub.load('jamesmcclain/algae-classifier:2a51273a16edaab22645f455e0b91002a74c702b',
                           'make_algae_model',
                           in_channels=[4, 12, 224],
                           backbone=args.backbone,
                           pretrained=False)
    model.load_state_dict(torch.load(args.pth_load))
    model.to(device)
    model.eval()

    if args.outfile is None:
        model_name = args.pth_load.split('/')[-1].split('.')[0]
        def transmute(filename):
            filename = filename.split('/')[-1]
            filename = f"./predict-{model_name}-{filename}"
            if not filename.endswith('.tiff'):
                filename = filename.replace('.tif', '.tiff')
            return filename
        args.outfile = [transmute(f) for f in args.infile]

    for (infile, outfile) in zip(args.infile, args.outfile):
        log.info(outfile)
        with rio.open(infile, 'r') as infile_ds, torch.no_grad():
            out_raw_profile = copy.deepcopy(infile_ds.profile)
            out_raw_profile.update({
                'compress': 'lzw',
                'dtype': np.float32,
                'count': 1,
                'bigtiff': 'yes',
                'sparse_ok': 'yes',
                'tiled': 'yes',
            })
            width = infile_ds.width
            height = infile_ds.height
            bandcount = infile_ds.count
            ar_out = torch.zeros((1, height, width), dtype=torch.float32).to(device)
            pixel_hits = torch.zeros((1, height, width), dtype=torch.uint8).to(device)

            if bandcount == 224:
                indexes = list(range(1, 224 + 1))
            elif bandcount in {12, 13}:
                indexes = list(range(1, 12 + 1))
                # NOTE: 13 bands does not indicate L1C support, this is
                # for Franklin COGs that have an extra band.
                bandcount = 12
            elif bandcount == 4:
                indexes = list(range(1, 4 + 1))
            elif bandcount == 5:
                indexes = [1, 2, 3, 5]
                bandcount = 4
            else:
                raise Exception(f'bands={bandcount}')

            # gather up batches
            batches = []
            for i in range(0, width - n, args.stride):
                for j in range(0, height - n, args.stride):
                    batches.append((i, j))
            batches = [batches[i:i + args.chunksize] for i in range(0, len(batches), args.chunksize)]

            for batch in tqdm.tqdm(batches):
                windows = [infile_ds.read(indexes, window=Window(i, j, n, n)) for (i, j) in batch]
                windows = [w.astype(np.float32) for w in windows]
                if args.ndwi_mask:
                    windows = [w * (((w[2] - w[7]) / (w[2] + w[7])) > 0.0) for w in windows]

                try:
                    windows = np.stack(windows, axis=0)
                except:
                    continue
                windows = torch.from_numpy(windows).to(dtype=torch.float32, device=device)
                prob = model(windows)

                for k, (i, j) in enumerate(batch):
                    if 'seg_out' in prob or args.force_class:
                        import pdb ; pdb.set_trace()
                        best = prob.get('seg_out').softmax(dim=1).detach().cpu().numpy()
                        best = np.argmax(best, axis=0)
                        ar_out[0, j:(j + n), i:(i + n)] += (best == 0)
                    else:
                        ar_out[0, j:(j + n), i:(i + n)] += torch.sigmoid(prob.get('cls_out').get('0')[k, 0])
                    pixel_hits[0, j:(j + n), i:(i + n)] += 1

        # Bring results back to CPU
        ar_out /= pixel_hits
        ar_out = ar_out.cpu().numpy()

        # Write results to file
        with rio.open(outfile, 'w', **out_raw_profile) as outfile_raw_ds:
            outfile_raw_ds.write(ar_out[0], indexes=1)
