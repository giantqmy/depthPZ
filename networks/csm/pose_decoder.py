# Copyright Niantic 2021. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the ManyDepth licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

import torch
import torch.nn as nn
from collections import OrderedDict


class PoseDecoder(nn.Module):
    def __init__(self, num_ch_enc, num_input_features, num_frames_to_predict_for=None, stride=1):
        super(PoseDecoder, self).__init__()

        self.num_ch_enc = num_ch_enc
        self.num_input_features = num_input_features

        if num_frames_to_predict_for is None:
            num_frames_to_predict_for = num_input_features - 1
        self.num_frames_to_predict_for = num_frames_to_predict_for

        self.convs = OrderedDict()
        self.convs[("squeeze")] = nn.Conv2d(self.num_ch_enc[-1], 256, 1)
        self.convs[("pose", 0)] = nn.Conv2d(257, 257, 3, stride, 1)
        self.convs[("pose", 1)] = nn.Conv2d(257, 257, 3, stride, 1)
        self.convs[("pose", 2)] = nn.Conv2d(257, 6 * num_frames_to_predict_for, 1)
        self.pool = torch.nn.MaxPool2d(stride*stride, stride=stride*stride)
        self.relu = nn.ReLU()
        # self.leaky_relu = nn.LeakyReLU(negative_slope=0.01)

        self.net = nn.ModuleList(list(self.convs.values()))

    def forward(self, input_features, mask, timedifference):
        last_features = [f[-1] for f in input_features]

        cat_features = [self.relu(self.convs["squeeze"](f)) for f in last_features]
        cat_features = torch.cat(cat_features, 1)

        out = cat_features
        out = torch.cat((out, torch.ones_like(out)[:, 0:1, :, :]*timedifference), 1)
        for i in range(3):
            out = self.convs[("pose", i)](out)
            if i != 2:
                out[:, -1, :, :] = torch.ones_like(out)[:, 0:1, :, :] * timedifference
                out = self.relu(out)

        mask = self.pool(mask).repeat(1, 6, 1, 1)

        masked_tensor = out * mask

        # Sum the masked tensor over dimensions 2 and 3
        sum_masked_tensor = masked_tensor.sum(dim=(2, 3))

        # Count the number of non-zero mask values for each channel
        count_non_zero = mask.sum(dim=(2, 3))

        # Compute the mean, taking care to avoid division by zero
        out = sum_masked_tensor / (count_non_zero + 1e-10)

        # out = out.mean(3).mean(2)

        out = 0.01 * out.view(-1, self.num_frames_to_predict_for, 1, 6)

        axisangle = out[..., :3]
        translation = out[..., 3:]

        axisangle = torch.tanh(axisangle) * 0.005
        translation_x = torch.tanh(translation)
        translation = translation_x.clone()
        translation[..., :2] = translation_x[..., :2] * 0.05
        translation[..., 2] = translation_x[..., 2] * 1.3
        # translation[:, :, :, 0:2] = torch.sigmoid(translation[:, :, :, 0:2]) * 0.1
        # translation[:, :, :, 2] = torch.sigmoid(translation[:, :, :, 2]) * 1.3


        # print(axisangle, translation)
        return axisangle, translation


