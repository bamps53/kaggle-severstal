import argparse
import os
import warnings

warnings.filterwarnings("ignore")

from catalyst.dl import SupervisedRunner
from catalyst.dl.callbacks import DiceCallback, IouCallback, CheckpointCallback, MixupCallback
import segmentation_models_pytorch as smp

from utils.config import load_config, save_config
from utils.callbacks import CutMixCallback
from datasets import make_loader
from optimizers import get_optimizer
from losses import get_loss
from schedulers import get_scheduler
from transforms import get_transforms


def run(config_file):
    config = load_config(config_file)

    if not os.path.exists(config.work_dir):
        os.makedirs(config.work_dir, exist_ok=True)
    save_config(config, config.work_dir + '/config.yml')

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    all_transforms = {}
    all_transforms['train'] = get_transforms(config.transforms.train)
    all_transforms['valid'] = get_transforms(config.transforms.test)

    dataloaders = {
        phase: make_loader(
            data_folder=config.data.train_dir,
            df_path=config.data.train_df_path,
            phase=phase,
            batch_size=config.train.batch_size,
            num_workers=config.num_workers,
            idx_fold=config.data.params.idx_fold,
            transforms=all_transforms[phase],
            num_classes=config.data.num_classes,
            pseudo_label_path=config.train.pseudo_label_path,
            debug=config.debug
        )
        for phase in ['train', 'valid']
    }

    # create segmentation model with pre trained encoder
    model = getattr(smp, config.model.arch)(
        encoder_name=config.model.encoder,
        encoder_weights=config.model.pretrained,
        classes=config.data.num_classes,
        activation=None,
    )

    # train setting
    criterion = get_loss(config)
    params = [
        {'params': model.decoder.parameters(), 'lr': config.optimizer.params.decoder_lr},
        {'params': model.encoder.parameters(), 'lr': config.optimizer.params.encoder_lr},
    ]
    optimizer = get_optimizer(params, config)
    scheduler = get_scheduler(optimizer, config)

    # model runner
    runner = SupervisedRunner(model=model)

    callbacks = [DiceCallback(), IouCallback()]

    # to resume from check points if exists
    if os.path.exists(config.work_dir + '/checkpoints/best.pth'):
        callbacks.append(CheckpointCallback(resume=config.work_dir + '/checkpoints/best_full.pth'))

    if config.train.mixup:
        callbacks.append(MixupCallback())

    if config.train.cutmix:
        callbacks.append(CutMixCallback())

    # model training
    runner.train(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        loaders=dataloaders,
        logdir=config.work_dir,
        num_epochs=config.train.num_epochs,
        callbacks=callbacks,
        verbose=True,
        fp16=True,
    )


def parse_args():
    parser = argparse.ArgumentParser(description='Severstal')
    parser.add_argument('--config', dest='config_file',
                        help='configuration file path',
                        default=None, type=str)
    return parser.parse_args()


def main():
    print('train Severstal Steel Defect Detection.')
    args = parse_args()
    if args.config_file is None:
        raise Exception('no configuration file')
    print('load config from {}'.format(args.config_file))
    run(args.config_file)


if __name__ == '__main__':
    main()
