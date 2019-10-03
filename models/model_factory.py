import pretrainedmodels
import torch
import torch.nn as nn
import torchvision


class CustomNet(nn.Module):
    def __init__(self, model_name, num_classes):
        super().__init__()
        self.model_name = model_name
        self.num_classes = num_classes

        if 'se_resnext' in self.model_name:
            self.net = getattr(pretrainedmodels, self.model_name)(pretrained='imagenet')
            self.net.avg_pool = nn.AdaptiveAvgPool2d(1)
            self.net.last_linear = nn.Linear(self.net.last_linear.in_features, self.num_classes)

        elif 'resnet' in self.model_name:
            self.net = torchvision.models.resnet50(pretrained=True)
            self.net.avg_pool = nn.AdaptiveAvgPool2d(1)
            self.net.fc = nn.Linear(self.net.fc.in_features, self.num_classes)

        elif self.model_name == 'resnext101_32x8d_wsl':
            self.net = torch.hub.load('facebookresearch/WSL-Images', 'resnext101_32x8d_wsl')
            self.net.fc = nn.Linear(self.net.fc.in_features, self.num_classes)

    def fresh_params(self):
        if 'se_resnext' in self.model_name:
            return self.net.last_linear.parameters()
        elif self.model_name == 'resnext101_32x8d_wsl':
            return self.net.fc.parameters()
        elif 'resnet' in self.model_name:
            return self.net.fc.parameters()

    def base_params(self):
        params = []

        if 'se_resnext' in self.model_name:
            fc_name = 'last_linear'
        elif self.model_name == 'resnext101_32x8d_wsl':
            fc_name = 'fc'
        elif 'resnet' in self.model_name:
            fc_name = 'fc'
        for name, param in self.net.named_parameters():
            if fc_name not in name:
                params.append(param)
        return params

    def forward(self, x):
        return self.net(x)
