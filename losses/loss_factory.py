import sys

sys.path.insert(0, '../..')
import torch
from torch.nn import functional as F

import torch.nn as nn
from . import functions

#from https://github.com/qubvel/segmentation_models.pytorch
class JaccardLoss(nn.Module):
    __name__ = 'jaccard_loss'

    def __init__(self, eps=1e-7, activation='sigmoid'):
        super().__init__()
        self.activation = activation
        self.eps = eps

    def forward(self, y_pr, y_gt):
        return 1 - functions.jaccard(y_pr, y_gt, eps=self.eps, threshold=None, activation=self.activation)


class DiceLoss(nn.Module):
    __name__ = 'dice_loss'

    def __init__(self, eps=1e-7, activation='sigmoid'):
        super().__init__()
        self.activation = activation
        self.eps = eps

    def forward(self, y_pr, y_gt):
        return 1 - functions.f_score(y_pr, y_gt, beta=1., eps=self.eps, threshold=None, activation=self.activation)


class CEDiceLoss(DiceLoss):
    __name__ = 'ce_dice_loss'

    def __init__(self, eps=1e-7, activation='softmax2d'):
        super().__init__(eps, activation)
        self.bce = nn.BCEWithLogitsLoss(reduction='mean')

    def forward(self, y_pr, y_gt):
        dice = super().forward(y_pr, y_gt)
        y_pr = torch.nn.Softmax2d()(y_pr)
        ce = self.bce(y_pr, y_gt)
        return dice + ce


class BCEJaccardLoss(JaccardLoss):
    __name__ = 'bce_jaccard_loss'

    def __init__(self, eps=1e-7, activation='sigmoid'):
        super().__init__(eps, activation)
        self.bce = nn.BCEWithLogitsLoss(reduction='mean')

    def forward(self, y_pr, y_gt):
        jaccard = super().forward(y_pr, y_gt)
        bce = self.bce(y_pr, y_gt)
        return jaccard + bce


class BCEDiceLoss(DiceLoss):
    __name__ = 'bce_dice_loss'

    def __init__(self, eps=1e-7, activation='sigmoid'):
        super().__init__(eps, activation)
        self.bce = nn.BCEWithLogitsLoss(reduction='mean')

    def forward(self, y_pr, y_gt):
        dice = super().forward(y_pr, y_gt)
        bce = self.bce(y_pr, y_gt)
        return dice + bce


class WeightedBCELoss(nn.Module):
    def __init__(self, pos_weight=0.75, neg_weight=0.25):
        super().__init__()
        self.pos_weight = pos_weight
        self.neg_weight = neg_weight

    def forward(self, logit, truth):
        batch_size, num_class, H, W = logit.shape
        logit = logit.view(batch_size, num_class)
        truth = truth.view(batch_size, num_class)
        assert (logit.shape == truth.shape)
        loss = F.binary_cross_entropy_with_logits(logit, truth, reduction='none')

        if weight is None:
            loss = loss.mean()

        else:
            pos = (truth > 0.5).float()
            neg = (truth < 0.5).float()
            pos_sum = pos.sum().item() + 1e-12
            neg_sum = neg.sum().item() + 1e-12
            loss = (self.pos_weight * pos * loss / pos_sum + self.neg_weight * neg * loss / neg_sum).sum()
            # raise NotImplementedError

        return loss


def get_loss(config):
    if config.loss.name == 'BCEDice':
        criterion = BCEDiceLoss(eps=1.)
    elif config.loss.name == 'CEDice':
        criterion = CEDiceLoss(eps=1.)
    elif config.loss == 'WeightedBCE':
        criterion = WeightedBCELoss()
    elif config.loss == 'BCE':
        criterion = nn.BCEWithLogitsLoss(reduction='mean')
    else:
        raise Exception('Your loss name is not implemented. Please choose from [BCEDice, CEDice, WeightedBCE, BCE]')
    return criterion
