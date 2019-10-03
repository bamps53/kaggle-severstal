import argparse
import json
import os
import warnings

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

warnings.filterwarnings("ignore")

from catalyst.dl import SupervisedRunner
from catalyst.dl.utils import load_checkpoint
import segmentation_models_pytorch as smp

from models import CustomNet
from utils.utils import mask2rle, post_process
from utils.config import load_config
from datasets import make_loader
from transforms import get_transforms


def run_cls(config_file_cls):
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    # ------------------------------------------------------------------------------------------------------------
    # 1. classification inference
    # ------------------------------------------------------------------------------------------------------------
    config = load_config(config_file_cls)

    model = CustomNet(config.model.encoder, config.data.num_classes)

    testloader = make_loader(
        data_folder=config.data.test_dir,
        df_path=config.data.sample_submission_path,
        phase='test',
        batch_size=config.test.batch_size,
        num_workers=config.num_workers,
        transforms=get_transforms(config.transforms.test)
    )

    model.to(config.device)
    model.eval()

    checkpoint = load_checkpoint(f"{config.work_dir}/checkpoints/best.pth")
    model.load_state_dict(checkpoint['model_state_dict'])

    all_fnames = []
    all_predictions = []
    with torch.no_grad():
        for i, (batch_fnames, batch_images) in enumerate(tqdm(testloader)):
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

            all_fnames.extend(batch_fnames)
            all_predictions.append(batch_preds)

    all_predictions = np.concatenate(all_predictions)

    np.save('all_preds', all_predictions)
    df = pd.DataFrame(data=all_predictions, index=all_fnames)

    df.to_csv('cls_preds.csv', index=False)
    df.to_csv(f"{config.work_dir}/cls_preds.csv", index=False)


def run_seg(config_file_seg):
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    # ------------------------------------------------------------------------------------------------------------
    # 2. segmentation inference
    # ------------------------------------------------------------------------------------------------------------
    config = load_config(config_file_seg)

    # create segmentation model with pretrained encoder
    model = getattr(smp, config.model.arch)(
        encoder_name=config.model.encoder,
        encoder_weights=config.model.pretrained,
        classes=config.data.num_classes,
        activation=None,
    )

    runner = SupervisedRunner(model=model)

    testloader = make_loader(
        data_folder=config.data.test_dir,
        df_path=config.data.sample_submission_path,
        phase='test',
        batch_size=config.test.batch_size,
        num_workers=config.num_workers,
        transforms=get_transforms(config.transforms.test)
    )

    model.to(config.device)
    model.eval()

    checkpoint = load_checkpoint(f"{config.work_dir}/checkpoints/best.pth")
    model.load_state_dict(checkpoint['model_state_dict'])

    if os.path.exists(config.work_dir + '/threshold_search.json'):
        with open(config.work_dir + '/threshold_search.json') as json_file:
            data = json.load(json_file)
        df = pd.DataFrame(data)
        min_sizes = list(df.T.idxmax().values.astype(int))
    else:
        min_sizes = config.test.min_size

    predictions = []
    with torch.no_grad():
        for i, (batch_fnames, batch_images) in enumerate(tqdm(testloader)):
            batch_preds = torch.sigmoid(model(batch_images.to(config.device)))

            if config.test.tta:
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
    df = pd.DataFrame(predictions, columns=['ImageId_ClassId', 'EncodedPixels'])
    df.to_csv(config.work_dir + "/submission.csv", index=False)


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
        print('classification inference Severstal Steel Defect Detection.')
        run_cls(args.config_file_cls)
    if args.config_file_seg != None:
        print('segmentation inference Severstal Steel Defect Detection.')
        run_seg(args.config_file_seg)


if __name__ == '__main__':
    main()
