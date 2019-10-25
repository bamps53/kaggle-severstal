import argparse
import os
import warnings

import numpy as np
import torch
from tqdm import tqdm

warnings.filterwarnings("ignore")

from catalyst.dl.utils import load_checkpoint
import segmentation_models_pytorch as smp

from models import CustomNet
from utils import predict_batch
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

    validloader = make_loader(
        data_folder=config.data.train_dir,
        df_path=config.data.train_df_path,
        phase='valid',
        batch_size=config.train.batch_size,
        num_workers=config.num_workers,
        idx_fold=config.data.params.idx_fold,
        transforms=get_transforms(config.transforms.test),
        num_classes=config.data.num_classes,
        task='cls'
    )

    model = CustomNet(config.model.encoder, config.data.num_classes)
    model.to(config.device)
    model.eval()
    checkpoint = load_checkpoint(f"{config.work_dir}/checkpoints/best.pth")
    model.load_state_dict(checkpoint['model_state_dict'])

    all_predictions = []
    all_targets = []
    with torch.no_grad():
        for i, (batch_images, batch_targets) in enumerate(tqdm(validloader)):
            batch_images = batch_images.to(config.device)
            batch_preds = predict_batch(model, batch_images, tta=config.test.tta, task='cls')

            all_targets.append(batch_targets)
            all_predictions.append(batch_preds)

    all_predictions = np.concatenate(all_predictions)
    all_targets = np.concatenate(all_targets)

    # evaluation
    all_accuracy_scores = []
    all_f1_scores = []
    thresholds = np.linspace(0.1, 0.9, 9)
    for th in thresholds:
        accuracy = accuracy_score(all_targets > th, all_predictions > th)
        f1 = f1_score(all_targets > th, all_predictions > th, average='samples')
        all_accuracy_scores.append(accuracy)
        all_f1_scores.append(f1)

    for th, score in zip(thresholds, all_accuracy_scores):
        print('validation accuracy for threshold {} = {}'.format(th, score))
    for th, score in zip(thresholds, all_f1_scores):
        print('validation f1 score for threshold {}  = {}'.format(th, score))

    np.save('valid_preds', all_predictions)


def run_seg(config_file_seg):
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    # ------------------------------------------------------------------------------------------------------------
    # 2. segmentation inference
    # ------------------------------------------------------------------------------------------------------------
    config = load_config(config_file_seg)

    validloader = make_loader(
        data_folder=config.data.train_dir,
        df_path=config.data.train_df_path,
        phase='valid',
        batch_size=config.train.batch_size,
        num_workers=config.num_workers,
        idx_fold=config.data.params.idx_fold,
        transforms=get_transforms(config.transforms.test),
        num_classes=config.data.num_classes,
    )

    # create segmentation model with pre-trained encoder
    model = getattr(smp, config.model.arch)(
        encoder_name=config.model.encoder,
        encoder_weights=config.model.pretrained,
        classes=config.data.num_classes,
        activation=None,
    )
    model.to(config.device)
    model.eval()
    checkpoint = load_checkpoint(f"{config.work_dir}/checkpoints/best.pth")
    model.load_state_dict(checkpoint['model_state_dict'])

    all_dice = {}
    min_sizes = [100, 300, 500, 750, 1000, 1500, 2000, 3000]
    for min_size in min_sizes:
        all_dice[min_size] = {}
        for cls in range(config.data.num_classes):
            all_dice[min_size][cls] = []

    with torch.no_grad():
        for i, (batch_images, batch_masks) in enumerate(tqdm(validloader)):
            batch_images = batch_images.to(config.device)
            batch_preds = predict_batch(model, batch_images, tta=config.test.tta)

            batch_masks = batch_masks.cpu().numpy()

            for masks, preds in zip(batch_masks, batch_preds):
                for cls in range(config.data.num_classes):
                    for min_size in min_sizes:
                        pred, _ = post_process(preds[cls, :, :], config.test.best_threshold, min_size)
                        mask = masks[cls, :, :]
                        all_dice[min_size][cls].append(dice_score(pred, mask))

    for cls in range(config.data.num_classes):
        for min_size in min_sizes:
            all_dice[min_size][cls] = sum(all_dice[min_size][cls]) / len(all_dice[min_size][cls])
            dict_to_json(all_dice, config.work_dir + '/threshold_search.json')
            if config.data.num_classes == 4:
                defect_class = cls + 1
            else:
                defect_class = cls
            print('average dice score for class{} for min_size {}: {}'.format(defect_class, min_size,
                                                                              all_dice[min_size][cls]))


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
