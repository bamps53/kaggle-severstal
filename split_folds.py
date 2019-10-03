import argparse
import os

import pandas as pd
from sklearn.model_selection import StratifiedKFold
from mlcomp.contrib.split import stratified_group_k_fold

from utils.config import load_config

def split_folds(config_file):
    df = pd.read_csv(config.data.train_df_path)
    df['exists'] = df['EncodedPixels'].notnull().astype(int)

    df['ImageId'] = df['ImageId_ClassId'].map(
        lambda x: x.split('_')[0].strip()
    )
    df['ClassId'] = df['ImageId_ClassId'].map(
        lambda x: int(x.split('_')[-1])
    )
    df['ClassId'] = [
        row.class_id if row.exists else 0 for row in df.itertuples()
    ]
    df['fold'] = stratified_group_k_fold(
        label='ClassId', group_column='ImageId', df=df, n_splits=config.data.params.num_folds
    )
    df.to_csv('folds.csv')


def parse_args():
    parser = argparse.ArgumentParser(description='Severstal')
    parser.add_argument('--config', dest='config_file',
                        help='configuration file path',
                        default=None, type=str)
    return parser.parse_args()


def main():
    print('split train dataset for Severstal Steel Defect Detection.')
    args = parse_args()
    if args.config_file is None:
        raise Exception('no configuration file')
    print('load config from {}'.format(args.config_file))
    split_folds(args.config_file)


if __name__ == '__main__':
    main()
