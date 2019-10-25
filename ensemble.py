import argparse
import json
import os
import warnings

import numpy as np
import pandas as pd
import torch
from pathlib import Path
from tqdm import tqdm

warnings.filterwarnings("ignore")

from models import MultiClsModels, MultiSegModels
from utils import predict_batch
from utils.utils import mask2rle, post_process, load_model
from utils.config import load_config
from datasets import make_loader
from transforms import get_transforms

KAGGLE_WORK_DIR = '/kaggle/working'


def run_cls(config_dir):
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    # ------------------------------------------------------------------------------------------------------------
    # 1. classification inference
    # ------------------------------------------------------------------------------------------------------------
    config_root = Path(config_dir) / 'cls'
    config_paths = [config_root / p for p in os.listdir(config_root)]
    base_config_paths = [Path(config_dir) / p for p in os.listdir(config_dir) if 'yml' in p]
    config = load_config(base_config_paths[0])

    models = []
    for c in config_paths:
        models.append(load_model(c))

    model = MultiClsModels(models)

    testloader = make_loader(
        data_folder=config.data.test_dir,
        df_path=config.data.sample_submission_path,
        phase='test',
        batch_size=config.test.batch_size,
        num_workers=config.num_workers,
        transforms=get_transforms(config.transforms.test),
        num_classes=config.data.num_classes,
    )

    all_fnames = []
    all_predictions = []
    with torch.no_grad():
        for i, (batch_fnames, batch_images) in enumerate(tqdm(testloader)):
            batch_images = batch_images.to(config.device)
            batch_preds = predict_batch(model, batch_images, tta=config.test.tta, task='cls')

            all_fnames.extend(batch_fnames)
            all_predictions.append(batch_preds)

    all_predictions = np.concatenate(all_predictions)

    np.save('all_preds', all_predictions)
    df = pd.DataFrame(data=all_predictions, index=all_fnames)

    df.to_csv('cls_preds.csv')
    df.to_csv(KAGGLE_WORK_DIR + '/cls_preds.csv')


def run_seg(config_dir):
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    # ------------------------------------------------------------------------------------------------------------
    # 2. segmentation inference
    # ------------------------------------------------------------------------------------------------------------
    config_root = Path(config_dir) / 'seg'
    config_paths = [config_root / p for p in os.listdir(config_root)]
    base_config_paths = [Path(config_dir) / p for p in os.listdir(config_dir) if 'yml' in p]
    config = load_config(base_config_paths[0])

    models = []
    for c in config_paths:
        models.append(load_model(c))

    model = MultiSegModels(models)

    if os.path.exists('cls_preds.csv'):
        testloader = make_loader(
            data_folder=config.data.test_dir,
            df_path='cls_preds.csv',
            phase='filtered_test',
            batch_size=config.test.batch_size,
            num_workers=config.num_workers,
            transforms=get_transforms(config.transforms.test)
        )
    else:
        testloader = make_loader(
            data_folder=config.data.test_dir,
            df_path=config.data.sample_submission_path,
            phase='test',
            batch_size=config.test.batch_size,
            num_workers=config.num_workers,
            transforms=get_transforms(config.transforms.test)
        )

    if os.path.exists(config.work_dir + '/threshold_search.json'):
        with open(config.work_dir + '/threshold_search.json') as json_file:
            data = json.load(json_file)
        df = pd.DataFrame(data)
        min_sizes = list(df.T.idxmax().values.astype(int))
        print('load best threshold from validation:', min_sizes)
    else:
        min_sizes = config.test.min_size
        print('load default threshold:', min_sizes)

    predictions = []
    with torch.no_grad():
        for i, (batch_fnames, batch_images) in enumerate(tqdm(testloader)):
            batch_images = batch_images.to(config.device)
            batch_preds = predict_batch(model, batch_images, tta=config.test.tta)

            for fname, preds in zip(batch_fnames, batch_preds):
                for cls in range(preds.shape[0]):
                    mask = preds[cls, :, :]
                    mask, num = post_process(mask, config.test.best_threshold, min_sizes[cls])
                    rle = mask2rle(mask)
                    name = fname + f"_{cls + 1}"
                    predictions.append([name, rle])

    # ------------------------------------------------------------------------------------------------------------
    # submission
    # ------------------------------------------------------------------------------------------------------------
    sub_df = pd.DataFrame(predictions, columns=['ImageId_ClassId', 'EncodedPixels'])

    sample_submission = pd.read_csv(config.data.sample_submission_path)
    df_merged = pd.merge(sample_submission, sub_df, on='ImageId_ClassId', how='left')
    df_merged.fillna('', inplace=True)
    df_merged['EncodedPixels'] = df_merged['EncodedPixels_y']
    df_merged = df_merged[['ImageId_ClassId', 'EncodedPixels']]

    df_merged.to_csv("submission.csv", index=False)
    df_merged.to_csv(KAGGLE_WORK_DIR + "/submission.csv", index=False)


def parse_args():
    parser = argparse.ArgumentParser(description='Severstal')
    parser.add_argument('--config_dir', default=None, type=str)
    return parser.parse_args()


def main():
    args = parse_args()
    run_cls(args.config_dir)
    run_seg(args.config_dir)


if __name__ == '__main__':
    main()
