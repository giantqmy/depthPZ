import torch
import torch.nn as nn
from .hrextractor import MPViT
from .hrextractor import UPerHead
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

def get_norm_cfg():
    return dict(type='BN', requires_grad=True)


class MPViTEncoder(nn.Module):
    def __init__(self, num_image_channels=3, lightweight=False, stride=2, upsample_fmaps=False, lidar_input=True,
                 out_channels=128,
                 ):

        super(MPViTEncoder, self).__init__()

        if lidar_input:
            in_channels = num_image_channels + 1 # + 1 lidar channel
        else:
            in_channels = num_image_channels

        self.in_channels = in_channels
        self.lightweight = lightweight

        norm_cfg = get_norm_cfg()

        self.mpvit = MPViT(
            num_stages=2 if self.lightweight else 4,
            in_chans=in_channels,
            num_path=[2, 3, 3, 3],
            num_layers=[1, 3, 6, 3],
            embed_dims=[64, 128, 216, 288],
            num_heads=[8, 8, 8, 8],
            mlp_ratios=[4, 4, 4, 4],
            drop_path_rate=0.2,
            norm_cfg=norm_cfg,
            norm_eval=False,
            stride=stride
        )

        self.neck = UPerHead(in_channels=[128, 128] if self.lightweight else [128, 216, 288, 288],
                             in_index=[0, 1] if self.lightweight else [0, 1, 2, 3],
                             pool_scales=(1, 2, 3, 6),
                             channels=256,
                             dropout_ratio=0.1,
                             num_classes=out_channels,
                             norm_cfg=norm_cfg,
                             align_corners=False,
                             loss_decode=dict(
                                 type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
                             upsample_fmaps=upsample_fmaps)

        self.mpvit.default_cfg = self._cfg_mpvit()


    def _cfg_mpvit(self, url="", **kwargs):
        return {
            "url": url,
            # "input_size": (self.in_channels, 224, 224),
            "pool_size": None,
            "crop_pct": 0.9,
            "interpolation": "bicubic",
            "mean": IMAGENET_DEFAULT_MEAN,
            "std": IMAGENET_DEFAULT_STD,
            "first_conv": "patch_embed.proj",
            "classifier": "head",
            **kwargs,
        }

    def forward(self, x):
        is_list = isinstance(x, tuple) or isinstance(x, list)
        if is_list:
            batch_dim = x[0].shape[0]
            x = torch.cat(x, dim=0)
        x = self.mpvit(x)
        x = self.neck(x)

        if is_list:
            x = x.split(split_size=batch_dim, dim=0)

        return x











