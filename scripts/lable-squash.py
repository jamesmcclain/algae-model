#!/usr/bin/env python3

import argparse
import glob
import json
import logging
import os
import os.path
import random
import sys

import numpy as np
import tqdm
from PIL import Image, PngImagePlugin


def cli_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--indir', required=True, type=str)
    parser.add_argument('--json', required=True, type=str)
    parser.add_argument('--outdir', required=True, type=str)
    return parser


if __name__ == '__main__':
    args = cli_parser().parse_args()
    logging.basicConfig(stream=sys.stderr, level=logging.INFO,
                        format='%(asctime)-15s %(message)s')
    log = logging.getLogger()

    with open(args.json, 'r') as f:
        groupings = json.load(f)

    for i, grouping in tqdm.tqdm(enumerate(groupings)):

        for nugget in grouping:
            npy_filename_in = f'{args.indir}/img/{nugget}'
            npy_filename_out = f'{args.outdir}/img/{i:04d}.npy'
            if os.path.exists(npy_filename_in) and not os.path.exists(npy_filename_out):
                os.system(f'ln "{npy_filename_in}" "{npy_filename_out}"')
                break

        labelss = []
        for nugget in grouping:
            png_filename = nugget.replace('.npy', '.png')
            png_filename_in = f'{args.indir}/labels/{png_filename}'
            if os.path.exists(png_filename_in):
                labelss.append(np.asarray(Image.open(png_filename_in)))
        if len(labelss) > 0:
            png_filename_out = f'{args.outdir}/labels/{i:04d}.png'

            diffs = []
            for one in labelss:
                diffs2 = []
                for two in labelss:
                    diffs2.append(one != two)
                diffs.append(np.sum(np.stack(diffs2), axis=0))

            usables = []
            for one in diffs:
                usables2 = []
                for two in diffs:
                    usables2.append(one > two)
                usables.append(np.sum(np.stack(usables2), axis=0))

            labels = np.copy(labelss[0])

            for i, usable in enumerate(usables):
                labels[usable == 0] = labelss[i][usable == 0]

            img = Image.fromarray(labels)
            img.save(png_filename_out)
