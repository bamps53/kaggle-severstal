from albumentations import (
    HorizontalFlip,
    VerticalFlip,
    RandomCrop,
    Resize,
    Cutout,
    Normalize,
    Compose,
    GaussNoise,
    IAAAdditiveGaussianNoise,
    RandomContrast,
    RandomGamma,
    RandomBrightness,
    OneOf)
from albumentations.pytorch import ToTensor

HEIGHT, WIDTH = 256, 1600


def get_transforms(phase_config):
    list_transforms = []
    if phase_config.HorizontalFlip:
        list_transforms.append(HorizontalFlip())
    if phase_config.VerticalFlip:
        list_transforms.append(VerticalFlip())
    if phase_config.RandomCropScale:
        list_transforms.extend(
            [RandomCrop(int(HEIGHT * 0.95), int(WIDTH * 0.95), p=0.5),
             Resize(HEIGHT, WIDTH, p=1)
             ]
        )
    if phase_config.Noise:
        list_transforms.append(
            OneOf([
                GaussNoise(),
                IAAAdditiveGaussianNoise(),
            ], p=0.5),
        )
    if phase_config.Contrast:
        list_transforms.append(
            OneOf([
                RandomContrast(0.5),
                RandomGamma(),
                RandomBrightness(),
            ], p=0.5),
        )

    if phase_config.RandomCropRotateScale:
        list_transforms.append(RandomCropRotateScale())
    if phase_config.Cutout.num_holes > 0:
        num_holes = phase_config.Cutout.num_holes
        hole_size = phase_config.Cutout.hole_size
        list_transforms.append(Cutout(num_holes, hole_size))
    if phase_config.CropSize > 0:
        list_transforms.append(RandomCrop(HEIGHT, phase_config.CropSize, p=1))

    list_transforms.extend(
        [
            Normalize(mean=phase_config.mean, std=phase_config.std, p=1),
            ToTensor(),
        ]
    )

    return Compose(list_transforms)
