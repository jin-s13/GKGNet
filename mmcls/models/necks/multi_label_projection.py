'''
用在GAP之后，将C维特征，通过num_class个FC变成num_class*d维向量
'''
# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn

from ..builder import NECKS


@NECKS.register_module()
class MultiLabelProjection(nn.Module):
    def __init__(self,
                 num_classes,
                 in_channels,
                 fea_d,
                 gap_dim):
        super(MultiLabelProjection, self).__init__()

        if num_classes <= 0:
            raise ValueError(
                f'num_classes={num_classes} must be a positive integer')

        self.in_channels = in_channels
        self.num_classes = num_classes
        self.fea_d=fea_d
        assert gap_dim in [1, 2, 3], 'GlobalAveragePooling gap_dim only support ' \
            f'{1, 2, 3}, get {gap_dim} instead.'
        if gap_dim == 1:
            self.gap = nn.AdaptiveAvgPool1d(1)
        elif gap_dim == 2:
            self.gap = nn.AdaptiveAvgPool2d((1, 1))
        else:
            self.gap = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc_list = nn.ModuleList()
        for i in range(self.num_classes):
            self.fc_list.append(
                nn.Sequential(
                # nn.BatchNorm1d(self.in_channels),
                nn.Linear(self.in_channels, self.fea_d))
            )

    def forward(self,inputs, **kwargs):
        if isinstance(inputs, tuple):
            # inputs=inputs[-1]
            outs = tuple([self.gap(x) for x in inputs])
            outs = tuple(
                [out.view(x.size(0), -1) for out, x in zip(outs, inputs)])
        elif isinstance(inputs, torch.Tensor):
            outs = self.gap(inputs)
            outs = outs.view(inputs.size(0), -1)
        else:
            raise TypeError('neck inputs should be tuple or torch.tensor')

        if isinstance(outs, tuple):
            x = outs[-1]
        else:
            x=outs

        batch_size=x.shape[0]
        features = []
        for i in range(self.num_classes):
            features.append(self.fc_list[i](x))
        # features = torch.cat(features)
        return features




