import warnings
from typing import List

import numpy as np
import torch


def water_mask(data, n):
    if n == 4:
        ndwi = (data[1] - data[3]) / (data[1] + data[3])
    elif n == 12:
        ndwi = (data[2] - data[7]) / (data[2] + data[7])
    elif n == 224:
        ndwi = (data[22] - data[50]) / (data[22] + data[50])
    else:
        raise Exception(n)
    return ndwi


def cloud_hack(data, n):
    if n == 4:
        not_cloud = ((data[2] > 900) * (data[2] < 4000))
    elif n == 12:
        not_cloud = ((data[3] > 100) * (data[3] < 1000))
    elif n == 224:
        not_cloud = ((data[33] > 600) * (data[33] < 2000))
    else:
        raise Exception(n)
    return not_cloud


def augment(data, n):
    if np.random.randint(0, 2) == 0:
        data = np.transpose(data, axes=(0, 2, 1))
    if np.random.randint(0, 2) == 0:
        data = np.flip(data, axis=(1 + np.random.randint(0, 2)))
    if np.random.randint(0, 2) == 0:
        data = np.transpose(data, (0, 2, 1))
    if np.random.randint(0, 5) < 1:
        data *= (1.0 + ((np.random.rand(n, 1, 1) - 0.5) / 50))
    if np.random.randint(0, 5) < 1:
        data *= (1.0 + ((np.random.rand(n, 32, 32) - 0.5) / 500))
    data = np.rot90(data, k=np.random.randint(0, 4), axes=(1, 2))
    if np.random.randint(0, 2) == 0:
        data = np.transpose(data, (0, 2, 1))
    return data.copy()


class AlgaeUnlabeledDataset(torch.utils.data.Dataset):
    def __init__(self,
                 savezs,
                 ndwi_mask: bool = False,
                 cloud_hack: bool = False,
                 augment: bool = False):
        self.yesno = []
        self.yesnos = []
        for savez in savezs:
            npz = np.load(savez)
            self.yesno.append(npz.get('yesno'))
            self.yesnos.append(self.yesno[-1].shape[-1])
        self.compound = list(zip(self.yesnos, self.yesno))
        self.ndwi_mask = ndwi_mask
        self.cloud_hack = cloud_hack
        self.augment = augment
        warnings.filterwarnings('ignore')

    def __len__(self):
        n = 0
        for yesnos in self.yesnos:
            n += yesnos
        return n

    def __getitem__(self, idx):
        for (yesnos, yesno) in self.compound:
            if idx < yesnos:
                data = yesno[..., idx]
                break
            idx -= yesnos

        data = data.transpose((2, 0, 1))
        n = data.shape[-3]

        # Water Mask
        if self.ndwi_mask:
            ndwi = water_mask(data, n)
            data *= (ndwi > 0.0)

        # Cloud Hack
        if self.cloud_hack:
            not_cloud = cloud_hack(data, n)
            data *= not_cloud

        # Augmentations
        if self.augment:
            data = augment(data, n)

        return data


class AlgaeClassificationDataset(torch.utils.data.Dataset):
    def __init__(self,
                 savezs: List[str],
                 ndwi_mask: bool = False,
                 cloud_hack: bool = False,
                 augment: bool = False):
        self.yes = []
        self.no = []
        self.yeas = []
        self.nays = []
        for savez in savezs:
            npz = np.load(savez)
            self.yes.append(npz.get('yes'))
            self.yeas.append(self.yes[-1].shape[-1])
            self.no.append(npz.get('no'))
            self.nays.append(self.no[-1].shape[-1])
        self.compound = list(
            zip(list(zip(self.yeas, self.nays)), list(zip(self.yes, self.no))))
        self.ndwi_mask = ndwi_mask
        self.cloud_hack = cloud_hack
        self.augment = augment
        warnings.filterwarnings('ignore')

    def __len__(self):
        n = 0
        for yeas in self.yeas:
            n += yeas
        for nays in self.nays:
            n += nays
        return n

    def __getitem__(self, idx):
        for ((yeas, nays), (yes, no)) in self.compound:
            if idx < yeas:
                data, label = yes[..., idx], 1
                break
            idx -= yeas
            if idx < nays:
                data, label = no[..., idx], 0
                break
            idx -= nays

        data = data.transpose((2, 0, 1))
        n = data.shape[-3]

        # Water Mask
        if self.ndwi_mask:
            ndwi = water_mask(data, n)
            data *= (ndwi > 0.0)

        # Cloud Hack
        if self.cloud_hack:
            not_cloud = cloud_hack(data, n)
            data *= not_cloud

        # Augmentations
        if self.augment:
            data = augment(data, n)

        return (data, label)