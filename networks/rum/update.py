import torch
import torch.nn as nn
import torch.nn.functional as F

# ------------------------- Flow Head -------------------------

class FlowHead(nn.Module):
    """Two-layer convolutional head for predicting 1-channel disparity deltas."""
    def __init__(self, input_dim=128, hidden_dim=256, out_channels=1):
        super().__init__()
        self.conv1 = nn.Conv2d(input_dim, hidden_dim, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, out_channels, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.conv2(self.relu(self.conv1(x)))


# ------------------------- SepConv GRU -------------------------

class SepConvGRU(nn.Module):
    """Spatially-separable convolutional GRU used in RAFT-style recurrent updates."""
    def __init__(self, hidden_dim=128, input_dim=192 + 128):
        super().__init__()
        self.convz1 = nn.Conv2d(hidden_dim + input_dim, hidden_dim, (1, 5), padding=(0, 2))
        self.convr1 = nn.Conv2d(hidden_dim + input_dim, hidden_dim, (1, 5), padding=(0, 2))
        self.convq1 = nn.Conv2d(hidden_dim + input_dim, hidden_dim, (1, 5), padding=(0, 2))

        self.convz2 = nn.Conv2d(hidden_dim + input_dim, hidden_dim, (5, 1), padding=(2, 0))
        self.convr2 = nn.Conv2d(hidden_dim + input_dim, hidden_dim, (5, 1), padding=(2, 0))
        self.convq2 = nn.Conv2d(hidden_dim + input_dim, hidden_dim, (5, 1), padding=(2, 0))

    def forward(self, h, x):
        # Horizontal update
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz1(hx))
        r = torch.sigmoid(self.convr1(hx))
        q = torch.tanh(self.convq1(torch.cat([r * h, x], dim=1)))
        h = (1 - z) * h + z * q

        # Vertical update
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz2(hx))
        r = torch.sigmoid(self.convr2(hx))
        q = torch.tanh(self.convq2(torch.cat([r * h, x], dim=1)))
        h = (1 - z) * h + z * q

        return h


# ------------------------- Motion Encoder -------------------------

class BasicMotionEncoder(nn.Module):
    """
    Encodes correlation volume and current (1-channel) disparity estimate
    into a 128-channel motion feature.
    """
    def __init__(self, cor_planes: int):
        super().__init__()
        # Correlation branch
        self.convc1 = nn.Conv2d(cor_planes, 256, 1)
        self.convc2 = nn.Conv2d(256, 192, 3, padding=1)

        # Disparity branch (1-channel input)
        self.convf1 = nn.Conv2d(1, 128, 7, padding=3)
        self.convf2 = nn.Conv2d(128, 64, 3, padding=1)

        # 64 (disp branch) + 192 (corr branch) = 256
        # We want total output 128 = 127 (conv) + 1 (raw disp)
        self.conv = nn.Conv2d(64 + 192, 127, 3, padding=1)

    def forward(self, flow: torch.Tensor, corr: torch.Tensor) -> torch.Tensor:
        """
        Args:
            flow : [B, 1, H, W]   (disparity in "flow units" for the current scale)
            corr : [B, cor_planes, H, W]

        Returns:
            motion_features : [B, 128, H, W]
        """
        cor = F.relu(self.convc1(corr))
        cor = F.relu(self.convc2(cor))

        flo = F.relu(self.convf1(flow))
        flo = F.relu(self.convf2(flo))

        fused = torch.cat([cor, flo], dim=1)
        out = F.relu(self.conv(fused))          # [B, 127, H, W]
        return torch.cat([out, flow], dim=1)    # [B, 128, H, W]


# ------------------------- RUM Update Block (NO LiDAR, NO confidence) -------------------------

class UpdateBlock(nn.Module):
    """
    Recurrent update module (RUM).

    Performs:
        - GRU state update (net)
        - Disparity refinement (1-channel)
        - RAFT upsampling mask prediction
    """
    def __init__(
        self,
        hidden_dim: int,
        cor_planes: int,
        mask_size: int = 8,
    ):
        super().__init__()

        self.encoder = BasicMotionEncoder(cor_planes)

        # GRU input includes: context (hidden_dim) + motion features (128)
        # inp is expected to have hidden_dim channels (same as your usage: inp=fmap features)
        gru_input_dim = hidden_dim + 128
        self.gru = SepConvGRU(hidden_dim=hidden_dim, input_dim=gru_input_dim)

        # 1-channel disparity head
        self.flow_head = FlowHead(hidden_dim, hidden_dim=256, out_channels=1)

        # RAFT upsampling mask (same as before)
        self.mask = nn.Sequential(
            nn.Conv2d(hidden_dim, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, mask_size ** 2 * 9, 1)
        )

    def forward(self, net: torch.Tensor, inp: torch.Tensor, corr: torch.Tensor, flow: torch.Tensor):
        """
        Args:
            net  : GRU hidden state            [B, hidden_dim, H, W]
            inp  : context/features            [B, hidden_dim, H, W]
            corr : correlation volume          [B, cor_planes, H, W]
            flow : previous disparity (1ch)    [B, 1, H, W]

        Returns:
            net_new  : updated GRU state
            flow_new : updated disparity (1ch)
            up_mask  : RAFT upsampling mask
        """
        motion_features = self.encoder(flow, corr)          # [B, 128, H, W]
        gru_inp = torch.cat((inp, motion_features), dim=1)  # [B, hidden_dim+128, H, W]

        net = self.gru(net, gru_inp)

        delta_flow = self.flow_head(net)                    # [B, 1, H, W]
        flow_new = flow + delta_flow

        up_mask = 0.25 * self.mask(net)

        return net, flow_new, up_mask
