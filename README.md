# Severstal: Steel Defect Detection
This is the code for 30th place solution in [kaggle Seversteal steel detection](https://www.kaggle.com/c/severstal-steel-defect-detection).

This might be a relatively simple approach. Just apply 5 class classification including background class and then 4 class segmenteation.  
For classification task, I trained resnet50, efficientnet-b3 and se-resnext50.  
For segmentation task, Unet with resnet18, PSPNet with resnet18 and FPN with resnet50.

