#!/usr/bin/env python3

import csv
import io
import itertools
import json
from copy import deepcopy
from datetime import datetime, timedelta

import lzip
import numpy as np
import pyproj
import rasterio as rio
import rasterio.crs
import rasterio.warp
import rasterio.windows
from shapely.geometry import MultiPoint, Point, mapping


def lzip_or_file(filename):
    if filename.endswith('.lz'):
        return io.StringIO(lzip.decompress_file(filename).decode('utf-8'))
    else:
        return open(filename, 'r')


def binary_search(ar, day):
    lo = 0
    hi = len(ar) - 1
    old = -1
    while (hi - lo) > 10:
        if abs(old - (hi - lo)) < 2:
            break
        old = hi - lo
        mid = old // 2
        if day <= ar[mid][0]:
            hi = mid
        else:
            lo = mid
    for i in range(lo, hi + 1):
        if i + 1 < len(ar) and ar[i][0] <= day and day <= ar[i + 1][0]:
            return i
    return -1


def day_search(ar, lo_day, hi_day):
    lo = binary_search(ar, lo_day)
    hi = binary_search(ar, hi_day) + 1
    while hi < len(ar) and ar[hi][0] <= hi_day:
        hi = hi + 1
    while lo < hi and ar[lo][0] < lo_day:
        lo = lo + 1
    # lo >= hi indicates failure
    return lo, hi


if __name__ == '__main__':
    import argparse

    def cli_parser():
        parser = argparse.ArgumentParser()
        parser.add_argument('--csv', required=True, type=str, nargs='+')
        parser.add_argument('--days', required=False, type=int, default=1)
        parser.add_argument('--json', required=True, type=str)
        parser.add_argument('--n', required=False, type=int, default=32)
        parser.add_argument('--savez', required=True, type=str)
        return parser

    args = cli_parser().parse_args()
    ddt = timedelta(days=args.days)
    wgs84 = rasterio.crs.CRS.from_epsg(4326)
    yes = []
    no = []

    # gather up observations and sort by date
    observations = []
    for filename in args.csv:
        with lzip_or_file(filename) as csv_file:
            csv_reader = csv.DictReader(csv_file)
            for _row in csv_reader:
                year = int(_row.get('year'))
                month = int(_row.get('month'))
                day = int(_row.get('day'))
                dt = datetime(year=year, month=month, day=day)
                lat = float(_row.get('lat'))
                lon = float(_row.get('lon'))
                xy = (lon, lat)
                cellcount = _row.get('cellcount')
                if cellcount == 'None':
                    cellcount = 1
                cellcount = int(cellcount)
                t = (dt, xy, cellcount)
                observations.append(t)
    observations = sorted(observations)

    # for each scene ...
    with lzip_or_file(args.json) as f:
        for feature in json.load(f).get('features'):
            try:
                dt = feature.get('properties').get('datetime').replace(
                    'Z', '+00:00')
                dt = datetime.fromisoformat(dt).replace(tzinfo=None)
                uri = feature.get('assets').get('tiff_0').get('href').replace(
                    's3://', '/vsis3/')
            except:
                continue

            if 'ardn' in uri:
                continue

            # open the scene, look for observations inside of it
            # (right date, right location) and extract
            with rio.open(uri, 'r') as ds:
                # data
                crs = deepcopy(ds.crs)
                w = ds.width
                h = ds.height

                # subset of observations
                lo, hi = day_search(observations, dt - ddt, dt + ddt)
                if lo >= hi:
                    continue
                observations2 = [observations[i] for i in range(lo, hi)]

                # x, y, and count data for each observation
                cs = [o[2] for o in observations2]
                xs = [o[1][0] for o in observations2]
                ys = [o[1][1] for o in observations2]
                (xs, ys) = rasterio.warp.transform(wgs84, crs, xs=xs, ys=ys)
                xys = [ds.index(x, y) for (x, y) in zip(xs, ys)]

                for ((x, y), c) in [(xy, c) for (xy, c) in zip(xys, cs)]:
                    if 0 <= x and x < w and 0 <= y and y < h:
                        print(f'{uri} {x} {y} {c}')
                        window = rasterio.windows.Window(x, y, args.n, args.n)
                        stuff = ds.read(window=window).transpose(
                            (1, 2, 0)).astype(np.float64)
                        stuff_shape = stuff.shape
                        desired_shape = (args.n, args.n, stuff_shape[2])
                        if stuff_shape != desired_shape:
                            temp = np.zeros(desired_shape, dtype=np.float64)
                            temp[0:stuff_shape[0], 0:stuff_shape[1], :] = stuff
                            stuff = temp
                        if (c > 0):
                            yes.append(stuff)
                        else:
                            no.append(stuff)

    # save results
    yes = np.stack(yes, axis=3)
    no = np.stack(no, axis=3)
    np.savez(args.savez, yes=yes, no=no)
