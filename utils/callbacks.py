import numpy as np
import torch
from catalyst.dl.callbacks import CriterionCallback
from catalyst.dl.core.state import RunnerState
from typing import List


class CutMixCallback(CriterionCallback):
    def __init__(
            self,
            fields: List[str] = ("features", "targets"),
            alpha=1.0,
            on_train_only=True,
            **kwargs
    ):
        assert len(fields) > 0, \
            "At least one field for CutMixCallback is required"
        assert alpha >= 0, "alpha must be>=0"

        super().__init__(**kwargs)

        self.on_train_only = on_train_only
        self.fields = fields
        self.alpha = alpha
        self.lam = 1
        self.index = None
        self.is_needed = True

    def on_loader_start(self, state: RunnerState):
        self.is_needed = not self.on_train_only or \
                         state.loader_name.startswith("train")

    def on_batch_start(self, state: RunnerState):
        if not self.is_needed:
            return

        if self.alpha > 0:
            self.lam = np.random.beta(self.alpha, self.alpha)
        else:
            self.lam = 1

        self.index = torch.randperm(state.input[self.fields[0]].shape[0])
        self.index.to(state.device)

        bbx1, bby1, bbx2, bby2 = rand_bbox(state.input[self.input_key].size(), self.lam)
        for f in self.fields:
            state.input[f][:, :, bbx1:bbx2, bby1:bby2] = state.input[f][self.index, :, bbx1:bbx2, bby1:bby2]


__all__ = ["CutMixCallback"]


def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2
