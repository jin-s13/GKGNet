# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn
from mmcv.cnn import ConvModule
from mmcv.runner import BaseModule
from torch.nn import Sequential as Seq
from ..builder import NECKS
from ..backbones.vig_model import Grapher, act_layer,GrapherLabel,GrapherFeatherLabel,GrapherFeature,GrapherFpn



@NECKS.register_module()
class ChannelMapper(BaseModule):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 conv_cfg=None,
                 norm_cfg=None,
                 img_scale=576,
                 act_cfg=dict(type='ReLU'),
                 num_outs=None,
                 init_cfg=dict(
                     type='Xavier', layer='Conv2d', distribution='uniform')):
        super(ChannelMapper, self).__init__(init_cfg)
        assert isinstance(in_channels, list)
        self.extra_convs = None
        if num_outs is None:
            num_outs = len(in_channels)
        self.convs = nn.ModuleList()
        for in_channel in in_channels:
            self.convs.append(
                ConvModule(
                    in_channel,
                    out_channels,
                    kernel_size,
                    padding=(kernel_size - 1) // 2,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg))
        if num_outs > len(in_channels):
            self.extra_convs = nn.ModuleList()
            for i in range(len(in_channels), num_outs):
                if i == len(in_channels):
                    in_channel = in_channels[-1]
                else:
                    in_channel = out_channels
                self.extra_convs.append(
                    ConvModule(
                        in_channel,
                        out_channels,
                        3,
                        stride=2,
                        padding=1,
                        conv_cfg=conv_cfg,
                        norm_cfg=norm_cfg,
                        act_cfg=act_cfg))
        self.fpn_inner_gcn=Seq(
            # GrapherFpn(in_channels, 9, 1, 'mr', 'gelu', 'batch',
            #          True, False, 0.2, r=1, n=(img_scale//8)**2, drop_path=0.1,
            #         relative_pos=True,norm_cfg=norm_cfg),
            GrapherFpn(out_channels, 9, 1, 'mr', 'gelu', 'batch',
                       True, False, 0.2, r=1, n=(img_scale // 16) ** 2, drop_path=0.1,
                       relative_pos=True,norm_cfg=norm_cfg),
            GrapherFpn(out_channels, 9, 1, 'mr', 'gelu', 'batch',
                       True, False, 0.2, r=1, n=(img_scale // 32) ** 2, drop_path=0.1,
                       relative_pos=True,norm_cfg=norm_cfg),
            # GrapherFpn(in_channels, 9, 1, 'mr', 'gelu', 'batch',
            #            True, False, 0.2, r=2, n=(img_scale//8)**2, drop_path=0.1,
            #            relative_pos=True,norm_cfg=norm_cfg),
            GrapherFpn(out_channels, 9, 1, 'mr', 'gelu', 'batch',
                       True, False, 0.2, r=2, n=(img_scale // 16) ** 2, drop_path=0.1,
                       relative_pos=True,norm_cfg=norm_cfg),
            )
        self.up_sample=ConvModule(
                    out_channels,
                    in_channels[-1],
                    1,
                    padding=0,
                    conv_cfg=None,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg)
    def forward(self, inputs):
        """Forward function."""
        assert len(inputs) == len(self.convs)
        outs = [self.convs[i](inputs[i]) for i in range(len(inputs))]
        if self.extra_convs:
            for i in range(len(self.extra_convs)):
                if i == 0:
                    outs.append(self.extra_convs[0](inputs[-1]))
                else:
                    outs.append(self.extra_convs[i](outs[-1]))

        outs.append(inputs[-1])

        outs[0]=self.fpn_inner_gcn[0](outs[0],outs[0])
        outs[1]=self.fpn_inner_gcn[1](outs[1],outs[1])
        # outs[2]=self.fpn_inner_gcn[2](outs[2],outs[2])

        outs[1]=self.fpn_inner_gcn[2](outs[1],outs[0])
        # outs[2]=self.fpn_inner_gcn[4](outs[2],outs[1])
        out_final=self.up_sample(outs[-2])+outs[-1]
        # inputs=self.up_sample(outs[2])



        return tuple([out_final])
