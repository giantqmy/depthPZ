"""Wavelet PIM from ultralytics/nn/modules/pz.py, adapted to stereo features.

The Haar, frequency fusion and dual-output PIM calculations match the
PZDualPIMStage core in ultralytics-hmoe-joint-scratch. Existing stereo
encoders provide the input features; no extra ConvNeXt stage is added.
"""

import torch
from torch import nn
import torch.nn.functional as F


class _HaarDWT(nn.Module):
    """Fixed per-channel Haar decomposition with padding for odd feature sizes."""

    def __init__(self, channels):
        super().__init__()
        filters = torch.tensor(
            [[[1.0, 1.0], [1.0, 1.0]], [[-1.0, -1.0], [1.0, 1.0]],
             [[-1.0, 1.0], [-1.0, 1.0]], [[1.0, -1.0], [-1.0, 1.0]]]
        ) / 2
        self.register_buffer("weight", filters[:, None].repeat(channels, 1, 1, 1))

    def forward(self, x):
        batch, channels, height, width = x.shape
        x = F.pad(x, (0, width % 2, 0, height % 2))
        bands = F.conv2d(x, self.weight.to(dtype=x.dtype), stride=2, groups=channels)
        bands = bands.reshape(batch, channels, 4, *bands.shape[-2:])
        return bands[:, :, 0], torch.cat(tuple(bands[:, :, i] for i in (1, 2, 3)), dim=1)


class _WaveletS0DoLPFusion(nn.Module):
    """Fuse intensity and DoLP independently in low and high Haar bands."""

    def __init__(self, channels):
        super().__init__()
        self.dwt = _HaarDWT(channels)

        def gate(in_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, channels, 1, bias=False), nn.BatchNorm2d(channels),
                nn.SiLU(inplace=True), nn.Conv2d(channels, 2, 1),
            )

        def proj(in_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, channels, 1, bias=False), nn.BatchNorm2d(channels), nn.SiLU(inplace=True)
            )

        self.low_gate = gate(channels * 2)
        self.high_gate = gate(channels * 6)
        self.high_proj = proj(channels * 3)
        self.out_proj = proj(channels * 2)

    def forward(self, rgb, dolp):
        low_rgb, high_rgb = self.dwt(rgb)
        low_dolp, high_dolp = self.dwt(dolp)
        low_w = self.low_gate(torch.cat((low_rgb, low_dolp), dim=1)).softmax(dim=1)
        high_w = self.high_gate(torch.cat((high_rgb, high_dolp), dim=1)).softmax(dim=1)
        low = low_rgb * low_w[:, :1] + low_dolp * low_w[:, 1:]
        high = self.high_proj(high_rgb * high_w[:, :1] + high_dolp * high_w[:, 1:])
        low = F.interpolate(low, size=rgb.shape[-2:], mode="bilinear", align_corners=False)
        high = F.interpolate(high, size=rgb.shape[-2:], mode="bilinear", align_corners=False)
        return self.out_proj(torch.cat((low, high), dim=1))


class _WaveletPIM(nn.Module):
    """PoolFormer branch refinement and shared wavelet residual, as in PZ PIM."""

    def __init__(self, channels):
        super().__init__()
        self.norm1_s0 = nn.BatchNorm2d(channels)
        self.norm1_dolp = nn.BatchNorm2d(channels)
        self.norm2_s0 = nn.BatchNorm2d(channels)
        self.norm2_dolp = nn.BatchNorm2d(channels)
        self.pool = nn.AvgPool2d(3, stride=1, padding=1, count_include_pad=False)
        self.mlp_s0 = nn.Sequential(
            nn.Conv2d(channels, channels * 4, 1), nn.GELU(), nn.Conv2d(channels * 4, channels, 1)
        )
        self.mlp_dolp = nn.Sequential(
            nn.Conv2d(channels, channels * 4, 1), nn.GELU(), nn.Conv2d(channels * 4, channels, 1)
        )
        self.wavelet_fusion = _WaveletS0DoLPFusion(channels)
        self.layer_scale_1_s0 = nn.Parameter(torch.full((channels,), 1e-5))
        self.layer_scale_1_dolp = nn.Parameter(torch.full((channels,), 1e-5))
        self.layer_scale_2_s0 = nn.Parameter(torch.full((channels,), 1e-5))
        self.layer_scale_2_dolp = nn.Parameter(torch.full((channels,), 1e-5))
        self.fusion_scale = nn.Parameter(torch.tensor(0.5))

    @staticmethod
    def _scale(gamma, x):
        return gamma.view(1, -1, 1, 1) * x

    def forward(self, rgb, dolp):
        n_rgb, n_dolp = self.norm1_s0(rgb), self.norm1_dolp(dolp)
        rgb = rgb + self._scale(self.layer_scale_1_s0, self.pool(n_rgb) - n_rgb)
        dolp = dolp + self._scale(self.layer_scale_1_dolp, self.pool(n_dolp) - n_dolp)
        n_rgb, n_dolp = self.norm2_s0(rgb), self.norm2_dolp(dolp)
        fusion = self.fusion_scale * self.wavelet_fusion(n_rgb, n_dolp)
        rgb = rgb + self._scale(self.layer_scale_2_s0, self.mlp_s0(n_rgb))
        dolp = dolp + self._scale(self.layer_scale_2_dolp, self.mlp_dolp(n_dolp))
        return rgb + fusion, dolp + fusion


class DOLPPIMFeatureFusion(nn.Module):
    """Dual-output PIM followed by a trainable 2C-to-C stereo readout.

    The readout initially averages the enhanced branches. PIM itself keeps
    the source initialization, including 1e-5 layer scales and fusion_scale=0.5.
    """

    def __init__(self, feature_channels=256):
        super().__init__()
        self.pim = _WaveletPIM(feature_channels)
        self.out_project = nn.Conv2d(2 * feature_channels, feature_channels, 1, bias=False)
        with torch.no_grad():
            self.out_project.weight.zero_()
            channels = torch.arange(feature_channels)
            self.out_project.weight[channels, channels, 0, 0] = 0.5
            self.out_project.weight[channels, channels + feature_channels, 0, 0] = 0.5

    def forward(self, image_feature, dolp_feature):
        image, dolp = self.pim(image_feature, dolp_feature)
        return self.out_project(torch.cat((image, dolp), dim=1))
