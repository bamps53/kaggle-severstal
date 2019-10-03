import os

import jpeg4py as jpeg
import pandas as pd
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from utils import make_mask


class TrainDataset(Dataset):
    def __init__(self, df, data_folder, phase, transforms):
        self.df = df
        self.root = data_folder
        self.phase = phase
        self.transforms = transforms
        self.fnames = self.df.index.tolist()

    def __getitem__(self, idx):
        image_id, mask = make_mask(idx, self.df)
        image_path = os.path.join(self.root, image_id)
        img = jpeg.JPEG(image_path).decode()
        augmented = self.transforms(image=img, mask=mask)
        img = augmented['image']
        mask = augmented['mask']  # 1x256x1600x4
        mask = mask[0].permute(2, 0, 1)  # 1x4x256x1600
        return img, mask, image_id

    def __len__(self):
        return len(self.fnames)


class ClsTrainDataset(Dataset):
    def __init__(self, df, data_folder, phase, transforms):
        self.df = df
        self.root = data_folder
        self.phase = phase
        self.transforms = transforms
        self.fnames = self.df.index.tolist()

    def __getitem__(self, idx):
        image_id = self.df.iloc[idx].name
        label = self.df.iloc[idx, :4].notnull().values.astype('f')
        image_path = os.path.join(self.root, image_id)
        img = jpeg.JPEG(image_path).decode()
        augmented = self.transforms(image=img)
        img = augmented['image']
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


def make_loader(
        data_folder,
        df_path,
        phase,
        batch_size=8,
        num_workers=2,
        idx_fold=None,
        transforms=None,
        task='seg'  # choice of ['cls', 'seg']
):
    df = pd.read_csv(df_path)
    if phase == 'test':
        image_dataset = TestDataset(data_folder, df, transforms)
        is_shuffle = False

    else:  # train or valid
        if os.path.exists('folds.csv'):
            folds = pd.read_csv('folds.csv', index_col='ImageId')
        else:
            raise Exception('You need to run split_folds.py beforehand.')

        if phase == "train":
            folds = folds[folds['fold'] != idx_fold]
            is_shuffle = True
        else:
            folds = folds[folds['fold'] == idx_fold]
            is_shuffle = False

        if task == 'seg':
            image_dataset = TrainDataset(folds, data_folder, phase, transforms)
        else:
            image_dataset = ClsTrainDataset(folds, data_folder, phase, transforms)

    return DataLoader(
        image_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        shuffle=is_shuffle,
    )
