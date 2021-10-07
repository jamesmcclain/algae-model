#!/usr/bin/env python3

import rasterio as rio
import copy
import numpy as np

if __name__ == '__main__':
    import argparse

    def cli_parser():
        parser = argparse.ArgumentParser()
        parser.add_argument('--infile', required=True, type=str, nargs='+')
        parser.add_argument('--outfile', required=False, default=None, type=str, nargs='+')
        parser.add_argument('--A', required=True, type=int)
        parser.add_argument('--B', required=True, type=int)
        return parser

    args = cli_parser().parse_args()

    if args.outfile is None:
        def transmute(filename):
            filename = filename.split('/')[-1]
            filename = f"./xdvi-{args.A}-{args.B}-{filename}"
            if not filename.endswith('.tiff'):
                filename = filename.replace('.tif', '.tiff')
            return filename
        args.outfile = [transmute(f) for f in args.infile]

    for (infile, outfile) in zip(args.infile, args.outfile):
        with rio.open(args.infile, 'r') as ds:
            profile = copy.copy(ds.profile)
            profile.update(compress='deflate',
                           dtype=np.float32,
                           count=1,
                           bigtiff='yes',
                           tiled='yes')
            width = ds.width
            height = ds.height
            out_data = np.zeros((1, height, width), dtype=np.float32)
            in_data = ds.read([args.A, args.B]).astype(np.float32)

        out_data[0] = (in_data[0] - in_data[1]) / (in_data[0] + in_data[1])

    with rio.open(args.outfile, 'w', **profile) as ds:
        ds.write(out_data)
