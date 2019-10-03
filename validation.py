import argparse
import os
import warnings

import numpy as np
import torch
from tqdm import tqdm as tqdm

warnings.filterwarnings("ignore")

from catalyst.dl.utils import load_checkpoint
import segmentation_models_pytorch as smp

from models import CustomNet
from utils.config import load_config
from utils.utils import post_process, dict_to_json
from utils.metrics import dice_score
from datasets import make_loader
from transforms import get_transforms

from sklearn.metrics import accuracy_score, f1_score


def run_cls(config_file_cls):
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    # ------------------------------------------------------------------------------------------------------------
    # 1. classification inference
    # ------------------------------------------------------------------------------------------------------------
    config = load_config(config_file_cls)

    model = CustomNet(config.model.encoder, config.data.num_classes)

    validloader = make_loader(
        data_folder=config.data.train_dir,
        df_path=config.data.train_df_path,
        phase='valid',
        batch_size=config.train.batch_size,
        num_workers=config.num_workers,
        idx_fold=config.data.params.idx_fold,
        transforms=get_transforms(config.transforms.test),
        task='cls'
    )

    model.to(config.device)
    model.eval()

    checkpoint = load_checkpoint(f"{config.work_dir}/checkpoints/best.pth")
    model.load_state_dict(checkpoint['model_state_dict'])

    all_predictions = []
    all_targets = []
    with torch.no_grad():
        for i, (batch_images, batch_targets) in enumerate(tqdm(validloader)):
            batch_preds = torch.sigmoid(model(batch_images.to(config.device)))

            # h_flip
            h_images = torch.flip(batch_images, dims=[3])
            batch_preds += torch.sigmoid(model(h_images.to(config.device)))

            # v_flip
            v_images = torch.flip(batch_images, dims=[2])
            batch_preds += torch.sigmoid(model(v_images.to(config.device)))

            # hv_flip
            hv_images = torch.flip(torch.flip(batch_images, dims=[3]), dims=[2])
            batch_preds += torch.sigmoid(model(hv_images.to(config.device)))

            batch_preds /= 4
            batch_preds = batch_preds.cpu().numpy()

            all_targets.append(batch_targets)
            all_predictions.append(batch_preds)

    all_predictions = np.concatenate(all_predictions)
    all_targets = np.concatenate(all_targets)

    for th in np.linspace(0.1, 0.9, 9):
        accuracy = accuracy_score(all_targets > th, all_predictions > th)
        f1 = f1_score(all_targets > th, all_predictions > th, average='samples')
        print('validation accuracy for threshold{} = {}'.format(th, accuracy))
        print('valiation f1 score for threshold{}  = {}'.format(th, f1))

    np.save('valid_preds', all_predictions)


def run_seg(config_file_seg):
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    # ------------------------------------------------------------------------------------------------------------
    # 2. segmentation inference
    # ------------------------------------------------------------------------------------------------------------
    config = load_config(config_file_seg)

    # create segmentation model with pre-trained encoder
    model = getattr(smp, config.model.arch)(
        encoder_name=config.model.encoder,
        encoder_weights=config.model.pretrained,
        classes=config.data.num_classes,
        activation=None,
    )

    validloader = make_loader(
        data_folder=config.data.train_dir,
        df_path=config.data.train_df_path,
        phase='valid',
        batch_size=config.train.batch_size,
        num_workers=config.num_workers,
        idx_fold=config.data.params.idx_fold,
        transforms=get_transforms(config.transforms.test)
    )

    model.to(config.device)
    model.eval()

    checkpoint = load_checkpoint(f"{config.work_dir}/checkpoints/best.pth")
    model.load_state_dict(checkpoint['model_state_dict'])

    all_dice = {}
    min_sizes = [250, 500, 750, 1000, 1500, 2000, 3000]
    for min_size in min_sizes:
        all_dice[min_size] = {}
        for cls in range(4):
            all_dice[min_size][cls] = []

    with torch.no_grad():
        for i, (batch_images, batch_masks) in enumerate(tqdm(validloader)):
            # default
            batch_preds = torch.sigmoid(model(batch_images.to(config.device)))

            # h_flip
            h_images = torch.flip(batch_images, dims=[3])
            h_batch_preds = torch.sigmoid(model(h_images.to(config.device)))
            batch_preds += torch.flip(h_batch_preds, dims=[3])

            # v_flip
            v_images = torch.flip(batch_images, dims=[2])
            v_batch_preds = torch.sigmoid(model(v_images.to(config.device)))
            batch_preds += torch.flip(v_batch_preds, dims=[2])

            # hv_flip
            hv_images = torch.flip(torch.flip(batch_images, dims=[3]), dims=[2])
            hv_batch_preds = torch.sigmoid(model(hv_images.to(config.device)))
            batch_preds += torch.flip(torch.flip(hv_batch_preds, dims=[3]), dims=[2])

            batch_preds /= 4
            batch_preds = batch_preds.detach().cpu().numpy()

            batch_masks.cpu().numpy()

            for masks, preds in zip(batch_masks, batch_preds):
                for cls in range(4):
                    for min_size in min_sizes:
                        pred, _ = post_process(preds[cls, :, :], config.test.best_threshold, min_size)
                        mask = masks[cls, :, :]
                        all_dice[min_size][cls].append(dice_score(pred, mask))

    for min_size in min_sizes:
        for cls in range(4):
            all_dice[min_size][cls] = sum(all_dice[min_size][cls]) / len(all_dice[min_size][cls])
            dict_to_json(all_dice, config.work_dir + '/threshold_search.json')
            print('average dice score for class{} for min_size {}: {}'.format(cls, min_size, all_dice[min_size][cls]))


def parse_args():
    parser = argparse.ArgumentParser(description='Severstal')
    parser.add_argument('--cls_config', dest='config_file_cls',
                        help='configuration file path',
                        default=None, type=str)
    parser.add_argument('--seg_config', dest='config_file_seg',
                        help='configuration file path',
                        default=None, type=str)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.config_file_cls != None:
        print('classification validation Severstal Steel Defect Detection.')
        run_cls(args.config_file_cls)
    if args.config_file_seg != None:
        print('segmentation validation Severstal Steel Defect Detection.')
        run_seg(args.config_file_seg)


if __name__ == '__main__':
    main()
