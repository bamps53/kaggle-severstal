import os

import jpeg4py as jpeg
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from utils import make_mask


class TrainDataset(Dataset):
    def __init__(self, df, data_folder, phase, transforms, num_classes, return_fnames):
        self.df = df
        self.root = data_folder
        self.phase = phase
        self.transforms = transforms
        self.fnames = self.df.index.tolist()
        self.num_classes = num_classes
        self.return_fnames = return_fnames

    def __getitem__(self, idx):
        image_id, mask = make_mask(idx, self.df)
        image_path = expand_path(image_id)
        img = jpeg.JPEG(str(image_path)).decode()
        augmented = self.transforms(image=img, mask=mask)
        img = augmented['image']
        mask = augmented['mask']  # 1x256x1600x4
        mask = mask[0].permute(2, 0, 1)  # 1x4x256x1600
        if self.num_classes == 5:
            mask_0 = (mask.sum(axis=0, keepdims=True) == 0).float()
            mask = torch.cat([mask_0, mask], axis=0)

        if self.return_fnames:
            return img, mask, image_id
        else:
            return img, mask

    def __len__(self):
        return len(self.fnames)


class ClsTrainDataset(Dataset):
    def __init__(self, df, data_folder, phase, transforms, num_classes=4, return_fnames=False):
        self.df = df
        self.root = data_folder
        self.phase = phase
        self.transforms = transforms
        self.fnames = self.df.index.tolist()
        self.num_classes = num_classes
        self.return_fnames = return_fnames

    def __getitem__(self, idx):
        image_id = self.df.iloc[idx].name
        if self.num_classes == 4:
            label = self.df.iloc[idx, :4].notnull().values.astype('f')
        else:
            label = np.zeros(5)
            label[1:5] = self.df.iloc[idx, :4].notnull()
            label[0] = label[1:5].sum() <= 0
            label = label.astype('f')

        image_path = os.path.join(self.root, image_id)
        img = jpeg.JPEG(image_path).decode()
        augmented = self.transforms(image=img)
        img = augmented['image']
        if self.return_fnames:
            return img, label, image_id
        else:
            return img, label

    def __len__(self):
        return len(self.fnames)


class TestDataset(Dataset):
    def __init__(self, root, df, transforms):
        self.root = root
        df['ImageId'] = df['ImageId_ClassId'].apply(lambda x: x.split('_')[0])
        self.fnames = df['ImageId'].unique().tolist()
        self.num_samples = len(self.fnames)
        self.transforms = transforms

    def __getitem__(self, idx):
        fname = self.fnames[idx]
        image_path = os.path.join(self.root, fname)
        img = jpeg.JPEG(image_path).decode()
        images = self.transforms(image=img)["image"]
        return fname, images

    def __len__(self):
        return self.num_samples


class FilteredTestDataset(Dataset):
    def __init__(self, root, df, transform):
        self.root = root
        df = df[(df > 0.5).sum(axis=1) > 0]  # screen no defect images
        self.fnames = df.index.tolist()
        self.num_samples = len(self.fnames)
        self.transform = transform

    def __getitem__(self, idx):
        fname = self.fnames[idx]
        image_path = os.path.join(self.root, fname)
        img = jpeg.JPEG(image_path).decode()
        images = self.transform(image=img)["image"]
        return fname, images

    def __len__(self):
        return self.num_samples


def expand_path(p):
    train_dir = Path('../input/severstal-steel-defect-detection/train_images')
    test_dir = Path('../input/severstal-steel-defect-detection/test_images')
    if (train_dir / p).exists():
        return train_dir / p
    elif (test_dir / p).exists():
        return test_dir / p


def make_loader(
        data_folder,
        df_path,
        phase,
        batch_size=8,
        num_workers=2,
        idx_fold=None,
        transforms=None,
        num_classes=4,
        pseudo_label_path=None,
        task='seg',  # choice of ['cls', 'seg'],
        return_fnames=False,
        debug=False,
):
    if debug:
        num_rows = 100
    else:
        num_rows = None

    df = pd.read_csv(df_path, nrows=num_rows)

    if phase == 'test':
        image_dataset = TestDataset(data_folder, df, transforms)
        is_shuffle = False

    elif phase == 'filtered_test':
        df = pd.read_csv(df_path, nrows=num_rows, index_col=0)
        image_dataset = FilteredTestDataset(data_folder, df, transforms)
        is_shuffle = False

    else:  # train or valid
        if os.path.exists('folds.csv'):
            folds = pd.read_csv('folds.csv', index_col='ImageId', nrows=num_rows)
        else:
            raise Exception('You need to run split_folds.py beforehand.')

        if phase == "train":
            folds = folds[folds['fold'] != idx_fold]

            if os.path.exists(pseudo_label_path):
                pseudo_df = pd.read_csv(pseudo_label_path)
                pseudo_df['ImageId'], pseudo_df['ClassId'] = zip(*pseudo_df['ImageId_ClassId'].str.split('_'))
                pseudo_df['ClassId'] = pseudo_df['ClassId'].astype(int)
                pseudo_df['exists'] = pseudo_df['EncodedPixels'].notnull().astype(int)
                pseudo_df['ClassId0'] = [row.ClassId if row.exists else 0 for row in pseudo_df.itertuples()]
                pv_df = pseudo_df.pivot(index='ImageId', columns='ClassId', values='EncodedPixels')
                folds = pd.concat([folds, pv_df], axis=0)

            is_shuffle = True
        else:
            folds = folds[folds['fold'] == idx_fold]
            is_shuffle = False

        if task == 'seg':
            image_dataset = TrainDataset(folds, data_folder, phase, transforms, num_classes, return_fnames)
        else:
            image_dataset = ClsTrainDataset(folds, data_folder, phase, transforms, num_classes, return_fnames)

    return DataLoader(
        image_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        shuffle=is_shuffle,
    )
