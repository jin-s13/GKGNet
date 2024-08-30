# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..builder import HEADS
from .cls_head import ClsHead
from ..losses.label_smooth_loss import LabelSmoothLoss

@HEADS.register_module()
class LabelQueryHead(ClsHead):
    """Linear classifier head.

    Args:
        num_classes (int): Number of categories excluding the background
            category.
        in_channels (int): Number of channels in the input feature map.
        init_cfg (dict | optional): The extra init config of layers.
            Defaults to use dict(type='Normal', layer='Linear', std=0.01).
    """

    def __init__(self,
                 num_classes,
                 in_channels,
                 softmax=  False,
                 double_loss=True,
                 init_cfg=dict(type='Normal', layer='Linear', std=0.01),
                 *args,
                 **kwargs):
        super(LabelQueryHead, self).__init__(init_cfg=init_cfg, *args, **kwargs)
        self.double_loss=double_loss
        self.bce_loss=LabelSmoothLoss(label_smooth_val=0.1, mode='multi_label',reduction='mean')
        self.in_channels = in_channels

        self.num_classes = num_classes
        self.softmax=softmax
        if self.num_classes <= 0:
            raise ValueError(
                f'num_classes={num_classes} must be a positive integer')

        self.fc1 = nn.Linear(self.in_channels, self.num_classes)
        self.fc2 = nn.Linear(self.in_channels, self.num_classes)

    def pre_logits(self, x):
        if isinstance(x, tuple):
            x = x[-1]
        return x

    def get_score(self,x):
        x0=x[0]
        x1=x[1]
        output1 = self.fc1(x0)
        diag_mask = torch.eye(output1.size(1)).unsqueeze(0).repeat(output1.size(0), 1, 1).cuda()
        output1 = (output1 * diag_mask).sum(-1)
        output3=self.fc2(x1)
        cls_score = output1+output3
        return cls_score
    def simple_test(self, x, softmax=False, post_process=True):
        cls_score = self.get_score(x)
        if self.softmax:
            pred = (
                F.softmax(cls_score, dim=1) if cls_score is not None else None)
        else:
            pred=(torch.sigmoid(cls_score) if cls_score is not None else None)
        if post_process:
            return self.post_process(pred)
        else:
            return pred

    def forward_train(self, x, gt_label, **kwargs):

        cls_score = self.get_score(x)

        _gt_label = torch.abs(gt_label)
        if self.softmax:
            gt_label=gt_label.view(-1).long()
        losses = self.loss(cls_score, gt_label, **kwargs)
        bce_loss=self.bce_loss(cls_score, gt_label, avg_factor=len(cls_score), **kwargs)
        if self.double_loss:
            return {
                'bce_loss':bce_loss,
                'asy_loss':losses['loss']*10.0
            }
        else:
            return losses

