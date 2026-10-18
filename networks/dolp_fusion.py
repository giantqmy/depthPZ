"""Feature-level DoLP fusion blocks for RGB/NIR stereo fine-tuning."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from networks.csm.attention_fusion import iAFF


class DOLPFeatureFusion(nn.Module):
    """Residual fusion that reuses a pretrained 128-channel iAFF block.

    The stereo encoders emit 256 channels, whereas the original CSM iAFF was
    trained on the 128-channel RUM context. Trainable projections initially
    select that context half. The zero-initialized expansion makes the block an identity
    at initialization, preserving the RGB/NIR checkpoint's starting output.
    """

    def __init__(self, feature_channels=256, fusion_channels=128):
        super().__init__()
        if feature_channels != 2 * fusion_channels:
            raise ValueError("DOLPFeatureFusion expects feature_channels == 2 * fusion_channels.")
        self.image_reduce = nn.Conv2d(feature_channels, fusion_channels, kernel_size=1, bias=False)
        self.dolp_reduce = nn.Conv2d(feature_channels, fusion_channels, kernel_size=1, bias=False)
        self.iaff = iAFF(channels=fusion_channels)
        self.expand = nn.Conv2d(fusion_channels, feature_channels, kernel_size=1, bias=False)

        with torch.no_grad():
            self.image_reduce.weight.zero_()
            self.dolp_reduce.weight.zero_()
            for channel in range(fusion_channels):
                self.image_reduce.weight[channel, channel + fusion_channels, 0, 0] = 1.0
                self.dolp_reduce.weight[channel, channel + fusion_channels, 0, 0] = 1.0
            self.expand.weight.zero_()

    def forward(self, image_feature, dolp_feature):
        image_context = F.relu(self.image_reduce(image_feature))
        dolp_context = F.relu(self.dolp_reduce(dolp_feature))
        fused_context = self.iaff(image_context, dolp_context)
        return image_feature + self.expand(fused_context)
