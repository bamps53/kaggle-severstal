import argparse
import os
import warnings

warnings.filterwarnings("ignore")

from catalyst.dl import SupervisedRunner
from catalyst.dl.callbacks import CheckpointCallback, F1ScoreCallback

from models import CustomNet

from utils.config import load_config, save_config
from datasets import make_loader
from optimizers import get_optimizer
from losses import get_loss
from schedulers import get_scheduler
from transforms import get_transforms


def run(config_file):
    config = load_config(config_file)

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
            task='cls'
        )
        for phase in ['train', 'valid']
    }

    # create model
    model = CustomNet(config.model.encoder, config.data.num_classes)

    # train setting
    criterion = get_loss(config)
    params = [
        {'params': model.base_params(), 'lr': config.optimizer.params.base_lr},
        {'params': model.fresh_params(), 'lr': config.optimizer.params.fresh_lr}
    ]
    optimizer = get_optimizer(params, config)
    scheduler = get_scheduler(optimizer, config)

    # model runner
    runner = SupervisedRunner(model=model)

    callbacks = [F1ScoreCallback()]
    if os.path.exists(config.work_dir + '/checkpoints/best.pth'):
        callbacks.append(CheckpointCallback(resume=config.work_dir + '/checkpoints/best_full.pth'))

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
                        help='configuration filename',
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
