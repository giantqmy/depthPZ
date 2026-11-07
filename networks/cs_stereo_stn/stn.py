"""STN module vendored from cs-stereo (MIT; see LICENSE)."""

import torch
import torch.nn as nn
import torch.nn.init as init

from .ednet import EDNet


def stn_init(module):
    if isinstance(module, nn.Conv2d):
        init.kaiming_normal_(module.weight.data)
        if module.bias is not None:
            init.constant_(module.bias.data, 1.0)


class STN(nn.Module):
    def __init__(self, in_shape, filt):
        super().__init__()
        self.ednet = EDNet(in_shape, 3, 3, 1.0 / 3.0)
        self.filt = filt

    def forward(self, rgbs, exp_ratio):
        if self.filt:
            filts = self.ednet(rgbs[0])
        else:
            filts = [torch.ones_like(rgbs[i]) / 3.0 for i in range(4)]
        return [
            (rgbs[i] * filts[i]).sum(1, keepdim=True) * exp_ratio
            for i in range(4)
        ]
