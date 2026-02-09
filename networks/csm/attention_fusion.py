import torch.nn as nn


class DAF(nn.Module):
    '''
    DirectAddFuse
    '''

    def __init__(self):
        super(DAF, self).__init__()

    def forward(self, x, residual):
        return x + residual


class iAFF(nn.Module):
    '''
    iAFF
    '''

    def __init__(self, channels=64, r=4, batch_size=1):
        super(iAFF, self).__init__()
        inter_channels = int(channels // r)


        # 本地注意力
        self.local_att = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        # 全局注意力
        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            # nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            # nn.BatchNorm2d(channels),
        )

        # 第二次本地注意力
        self.local_att2 = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )
        # 第二次全局注意力
        self.global_att2 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            # nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            # nn.BatchNorm2d(channels),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x, residual):
        xl_1 = self.local_att(x)
        xl_2 = self.local_att(residual)
        xg_1 = self.global_att(x)
        xg_2 = self.global_att(residual)
        xlg_1 = xl_1 + xg_1
        xlg_2 = xl_2 + xg_2
        wei_1 = self.sigmoid(xlg_1)
        wei_2 = self.sigmoid(xlg_2)
        total_weight = wei_1 + wei_2
        wei_1_scaled = wei_1/total_weight
        wei_2_scaled = wei_2/total_weight

        xi = 2 * x * wei_1_scaled + 2 * residual * wei_2_scaled

        xl2 = self.local_att2(xi)
        xg2 = self.global_att(xi)
        xlg2 = xl2 + xg2
        wei2 = self.sigmoid(xlg2)
        xo = 2*x*wei_1_scaled * wei2 + 2*residual*wei_2_scaled * (1-wei2)
        return xo


"""class AFF(nn.Module):
    '''
    AFF
    '''

    def __init__(self, channels=64, r=4):
        super(AFF, self).__init__()
        inter_channels = int(channels // r)

        self.local_att = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            # nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            # nn.BatchNorm2d(channels),
        )

        self.cross_att = LocalFeatureTransformer(
            d_model=128, nhead=8, layer_names=["cross"] * 1, attention="linear"
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x, residual):
        N, C, H, W = x.shape

        x = x.permute(0, 2, 3, 1).reshape(N, H * W, C)
        residual = residual.permute(0, 2, 3, 1).reshape(N, H * W, C)
        x, residual = self.cross_att(x, residual)
        x, residual = [
            x.reshape(N, H, W, C).permute(0, 3, 1, 2)
            for x in [x, residual]
        ]

        xa = x + residual
        xl = self.local_att(xa)
        xg = self.global_att(xa)
        xlg = xl + xg
        wei = self.sigmoid(xlg)
        xo = x * wei + residual * (1 - wei)

        # xa = x + residual
        # xl_1 = self.local_att(xa)
        # # xl_2 = self.local_att(residual)
        # xg = self.global_att(xa)
        # xlg_1 = xl_1 + xg
        # xlg_2 = xl_2 + xg
        # wei_1 = self.sigmoid(xlg_1)
        # wei_2 = self.sigmoid(xlg_2)
        #
        # xo = 2 * x * wei_1 + 2 * residual * wei_2
        return xo"""


class MS_CAM(nn.Module):

    def __init__(self, channels=64, r=4):
        super(MS_CAM, self).__init__()
        inter_channels = int(channels // r)

        self.local_att = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        xl = self.local_att(x)
        xg = self.global_att(x)
        xlg = xl + xg
        wei = self.sigmoid(xlg)
        return x * wei

