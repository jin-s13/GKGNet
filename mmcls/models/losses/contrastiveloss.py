# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn
import torch.nn.functional as F

from ..builder import LOSSES
from .utils import weight_reduce_loss
import torch

@LOSSES.register_module()
class ContrastiveLoss(torch.nn.Module):
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """
    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, pre_relation, gt_relation,**kwargs):
        losses=0.5*torch.mean((gt_relation.float()*torch.pow(pre_relation,2))) +\
               0.5*torch.mean((torch.pow(torch.clamp(2 - pre_relation, min=0.0), 2)) * (1 - gt_relation.float()))
        return losses