# Severstal: Steel Defect Detection
This is the code for 30th place solution in [kaggle Seversteal steel detection](https://www.kaggle.com/c/severstal-steel-defect-detection).

This might be a relatively simple approach. Just apply 5 class classification including background class and then 4 class segmenteation.  
For classification task, I trained resnet50, efficientnet-b3 and se-resnext50.  
For segmentation task, Unet with resnet18, PSPNet with resnet18 and FPN with resnet50.

## Usage

```
$ python split_fold.py --config config/base_config.yml
$ python train_cls.py --config config/cls/001_resnet50_BCE_5class_fold0.yml
$ python train_cls.py --config config/cls/002_efnet_b3_cls_BCE_5class_fold1.yml
$ python train_cls.py --config config/cls/003_seresnext50_cls_BCE_5class_fold2.yml
$ python train_seg.py --config config/seg/001_resnet18_Unet_fold0.yml
$ python train_seg.py --config config/seg/002_resnet18_PSPNet_fold0.yml
$ python train_seg.py --config config/seg/003_resnet50_fpn_fold0.yml
$ python ensemble.py --config_dir config/
```

## Reference

My code for this competition specific part is based on [this great starter kernel](https://www.kaggle.com/rishabhiitbhu/unet-starter-kernel-pytorch-lb-0-88)  

[this kernel](https://www.kaggle.com/lightforever/severstal-mlcomp-catalyst-infer-0-90672) also inspired me a lot.  

And I borrowed many idea from Heng's great disscussion topics.
