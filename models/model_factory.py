import pretrainedmodels
import torch
import torch.nn as nn
import torchvision
from efficientnet_pytorch import EfficientNet


class CustomNet(nn.Module):
    def __init__(self, model_name, num_classes, pretrained=True):
        super().__init__()
        self.model_name = model_name
        self.num_classes = num_classes
        self.pretrained = pretrained

        if 'se_resnext' in self.model_name:
            # weights_path = '../input/pytorch-pretrained-models/se_resnext101_32x4d-3b2fe3d8.pth'
            self.net = getattr(pretrainedmodels, self.model_name)(pretrained='imagenet')
            # self.net.load_state_dict(torch.load(weights_path))
            self.net.avg_pool = nn.AdaptiveAvgPool2d(1)
            self.net.last_linear = nn.Linear(self.net.last_linear.in_features, self.num_classes)

        elif 'resnet' in self.model_name:
            self.net = getattr(torchvision.models, self.model_name)(pretrained=True)
            self.net.avg_pool = nn.AdaptiveAvgPool2d(1)
            self.net.fc = nn.Linear(self.net.fc.in_features, self.num_classes)

        elif 'efficientnet' in self.model_name:
            if pretrained:
                self.net = EfficientNet.from_pretrained(self.model_name)
            else:
                self.net = EfficientNet.from_name(self.model_name)
            self.net._fc = nn.Linear(self.net._fc.in_features, self.num_classes)

    def fresh_params(self):
        if 'se_resnext' in self.model_name:
            return self.net.last_linear.parameters()
        elif 'resnet' in self.model_name:
            return self.net.fc.parameters()
        elif 'efficientnet' in self.model_name:
            return self.net._fc.parameters()

    def base_params(self):
        params = []

        if 'se_resnext' in self.model_name:
            fc_name = 'last_linear'
        elif 'resnet' in self.model_name:
            fc_name = 'fc'
        elif 'efficientnet' in self.model_name:
            fc_name = '_fc'
        for name, param in self.net.named_parameters():
            if fc_name not in name:
                params.append(param)
        return params

    def forward(self, x):
        return self.net(x)


class MultiClsModels:
    def __init__(self, models):
        self.models = models

    def __call__(self, x):
        res = []
        x = x.cuda()
        with torch.no_grad():
            for m in self.models:
                res.append(m(x)[:, -4:])
        res = torch.stack(res)
        return torch.mean(res, dim=0)


class MultiSegModels:
    def __init__(self, models):
        self.models = models

    def __call__(self, x):
        res = []
        x = x.cuda()
        with torch.no_grad():
            for m in self.models:
                res.append(m(x)[:, -4:, :, :])
        res = torch.stack(res)
        return torch.mean(res, dim=0)
