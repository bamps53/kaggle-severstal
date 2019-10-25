import yaml
from easydict import EasyDict as edict


def _get_default_config():
    c = edict()

    # dataset
    c.data = edict()
    c.data.name = 'DefaultDataset'
    c.data.num_classes = 4
    c.data.sample_submission_path = '../input/severstal-steel-defect-detection/sample_submission.csv'
    c.data.test_dir = '../input/severstal-steel-defect-detection/test_images'
    c.data.train_df_path = '../input/severstal-steel-defect-detection/train.csv'
    c.data.train_dir = '../input/severstal-steel-defect-detection/train_images'
    c.data.params = edict()

    # model
    c.model = edict()
    c.model.arch = 'Unet'
    c.model.encoder = 'resnet18'
    c.model.pretrained = 'imagenet'
    c.model.params = edict()

    # train
    c.train = edict()
    c.train.batch_size = 32
    c.train.num_epochs = 50
    c.train.mixup = False
    c.train.cutmix = False
    c.train.pseudo_label_path = 'dummy.csv'

    # test
    c.test = edict()
    c.test.batch_size = 32
    c.test.best_threshold = 0.5
    c.test.min_size = 3500
    c.test.tta = True

    # optimizer
    c.optimizer = edict()
    c.optimizer.name = 'Adam'
    c.optimizer.params = edict()

    # scheduler
    c.scheduler = edict()
    c.scheduler.name = 'plateau'
    c.scheduler.params = edict()

    # transforms
    c.transforms = edict()
    c.transforms.params = edict()

    c.transforms.train = edict()
    c.transforms.train.HorizontalFlip = True
    c.transforms.train.VerticalFlip = True
    c.transforms.train.RandomCropScale = False
    c.transforms.train.RandomCropRotateScale = False
    c.transforms.train.Cutout = edict()
    c.transforms.train.Cutout.num_holes = 0
    c.transforms.train.Cutout.hole_size = 25
    c.transforms.train.CropSize = 0
    c.transforms.train.mean = [0.485, 0.456, 0.406]
    c.transforms.train.std = [0.229, 0.224, 0.225]
    c.transforms.train.Contrast = False
    c.transforms.train.Noise = False

    c.transforms.test = edict()
    c.transforms.test.HorizontalFlip = False
    c.transforms.test.VerticalFlip = False
    c.transforms.test.RandomCropScale = False
    c.transforms.test.RandomCropRotateScale = False
    c.transforms.test.Cutout = edict()
    c.transforms.test.Cutout.num_holes = 0
    c.transforms.test.Cutout.hole_size = 25
    c.transforms.test.CropSize = 0
    c.transforms.test.mean = [0.485, 0.456, 0.406]
    c.transforms.test.std = [0.229, 0.224, 0.225]
    c.transforms.test.Contrast = False
    c.transforms.test.Noise = False

    # losses
    c.loss = edict()
    c.loss.name = 'BCEDice'
    c.loss.params = edict()

    c.device = 'cuda'
    c.num_workers = 2
    c.work_dir = './work_dir'
    c.checkpoint_path = './checkpoints/best.pth'
    c.debug = False

    return c


def _merge_config(src, dst):
    if not isinstance(src, edict):
        return

    for k, v in src.items():
        if isinstance(v, edict):
            _merge_config(src[k], dst[k])
        else:
            dst[k] = v


def load_config(config_path):
    with open(config_path, 'r') as fid:
        yaml_config = edict(yaml.load(fid, Loader=yaml.SafeLoader))

    config = _get_default_config()
    _merge_config(yaml_config, config)

    return config


def save_config(config, file_name):
    with open(file_name, "w") as wf:
        yaml.dump(config, wf)
