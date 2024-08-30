# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn
import torch.nn.functional as F

from ..builder import LOSSES
from .utils import weight_reduce_loss
import torch

def cross_entropy(pred,
                  label,
                  weight=None,
                  reduction='mean',
                  avg_factor=None,
                  class_weight=None,**kwargs):
    """Calculate the CrossEntropy loss.

    Args:
        pred (torch.Tensor): The prediction with shape (N, C), C is the number
            of classes.
        label (torch.Tensor): The gt label of the prediction.
        weight (torch.Tensor, optional): Sample-wise loss weight.
        reduction (str): The method used to reduce the loss.
        avg_factor (int, optional): Average factor that is used to average
            the loss. Defaults to None.
        class_weight (torch.Tensor, optional): The weight for each class with
            shape (C), C is the number of classes. Default None.

    Returns:
        torch.Tensor: The calculated loss
    """
    # element-wise losses
    loss = F.cross_entropy(pred, label, weight=class_weight, reduction='none')

    # apply weights and do the reduction
    if weight is not None:
        weight = weight.float()
    loss = weight_reduce_loss(
        loss, weight=weight, reduction=reduction, avg_factor=avg_factor)

    return loss


def soft_cross_entropy(pred,
                       label,
                       weight=None,
                       reduction='mean',
                       class_weight=None,
                       avg_factor=None,**kwargs):
    """Calculate the Soft CrossEntropy loss. The label can be float.

    Args:
        pred (torch.Tensor): The prediction with shape (N, C), C is the number
            of classes.
        label (torch.Tensor): The gt label of the prediction with shape (N, C).
            When using "mixup", the label can be float.
        weight (torch.Tensor, optional): Sample-wise loss weight.
        reduction (str): The method used to reduce the loss.
        avg_factor (int, optional): Average factor that is used to average
            the loss. Defaults to None.
        class_weight (torch.Tensor, optional): The weight for each class with
            shape (C), C is the number of classes. Default None.

    Returns:
        torch.Tensor: The calculated loss
    """
    # element-wise losses
    loss = -label * F.log_softmax(pred, dim=-1)
    if class_weight is not None:
        loss *= class_weight
    loss = loss.sum(dim=-1)

    # apply weights and do the reduction
    if weight is not None:
        weight = weight.float()
    loss = weight_reduce_loss(
        loss, weight=weight, reduction=reduction, avg_factor=avg_factor)

    return loss


def binary_cross_entropy(pred,
                         label,
                         weight=None,
                         reduction='mean',
                         avg_factor=None,
                         class_weight=None,
                         pos_weight=None,**kwargs):
    r"""Calculate the binary CrossEntropy loss with logits.

    Args:
        pred (torch.Tensor): The prediction with shape (N, \*).
        label (torch.Tensor): The gt label with shape (N, \*).
        weight (torch.Tensor, optional): Element-wise weight of loss with shape
            (N, ). Defaults to None.
        reduction (str): The method used to reduce the loss.
            Options are "none", "mean" and "sum". If reduction is 'none' , loss
            is same shape as pred and label. Defaults to 'mean'.
        avg_factor (int, optional): Average factor that is used to average
            the loss. Defaults to None.
        class_weight (torch.Tensor, optional): The weight for each class with
            shape (C), C is the number of classes. Default None.
        pos_weight (torch.Tensor, optional): The positive weight for each
            class with shape (C), C is the number of classes. Default None.

    Returns:
        torch.Tensor: The calculated loss
    """
    # Ensure that the size of class_weight is consistent with pred and label to
    # avoid automatic boracast,
    assert pred.dim() == label.dim()

    if class_weight is not None:
        N = pred.size()[0]
        class_weight = class_weight.repeat(N, 1)
    loss = F.binary_cross_entropy_with_logits(
        pred,
        label,
        weight=class_weight,
        pos_weight=pos_weight,
        reduction='none')
    # 添加loss权重
    # --------------------- AAAI ---------------------
    # pos_weights = torch.sqrt(1 / (2 * ratio.sqrt())) * targets
    # neg_weights = torch.sqrt(1 / (2 * (1 - ratio.sqrt()))) * (1 - targets)
    # weights = pos_weights + neg_weights
    if 'label_rate' in kwargs :
        # print('have label_rate-')
        label_rate=kwargs['label_rate']
        # pos_weights = label * (1 - label_rate)
        # neg_weights = (1 - label) * label_rate
        # weights = torch.exp(neg_weights + pos_weights)

        pos_weights = label * torch.sqrt(1/2.0/label_rate)
        neg_weights = (1 - label) * torch.sqrt(1/2.0/(1-label_rate))
        weights_cate = neg_weights + pos_weights
        #
        # # ACSL
        # weights=label.clone()
        # weights[label>0.5]=1
        # weights[label<0.5]=0
        # weights[(label<0.5) * (torch.sigmoid(pred)>0.3)]=1

        # weights=weights*weights_cate
        weights=weights_cate


        weights[label > 1] = 0.0

        loss=loss*weights
    # apply weights and do the reduction
    if weight is not None:
        assert weight.dim() == 1
        weight = weight.float()
        if pred.dim() > 1:
            weight = weight.reshape(-1, 1)
    loss = weight_reduce_loss(
        loss, weight=weight, reduction=reduction, avg_factor=avg_factor)
    return loss

def balanced_softmax_loss(pred,label, sample_per_class=None,
                          weight=None,
                         reduction='mean',
                         avg_factor=None,
                         class_weight=None,
                         pos_weight=None,**kwargs):
    """Compute the Balanced Softmax Loss between `pred` and the ground truth `labels`.
    """
    if sample_per_class is None:
        sample_per_class=kwargs['label_rate'][-1]
    spc = sample_per_class.type_as(pred)
    spc = spc.unsqueeze(0).expand(pred.shape[0], -1)
    pred = pred + spc.log()
    # element-wise losses
    loss = -label * F.log_softmax(pred, dim=-1)
    if class_weight is not None:
        loss *= class_weight
    loss = loss.sum(dim=-1)

    # apply weights and do the reduction
    if weight is not None:
        weight = weight.float()
    loss = weight_reduce_loss(
        loss, weight=weight, reduction=reduction, avg_factor=avg_factor)

    return loss
@LOSSES.register_module()
class CrossEntropyLoss(nn.Module):
    """Cross entropy loss.

    Args:
        use_sigmoid (bool): Whether the prediction uses sigmoid
            of softmax. Defaults to False.
        use_soft (bool): Whether to use the soft version of CrossEntropyLoss.
            Defaults to False.
        reduction (str): The method used to reduce the loss.
            Options are "none", "mean" and "sum". Defaults to 'mean'.
        loss_weight (float):  Weight of the loss. Defaults to 1.0.
        class_weight (List[float], optional): The weight for each class with
            shape (C), C is the number of classes. Default None.
        pos_weight (List[float], optional): The positive weight for each
            class with shape (C), C is the number of classes. Only enabled in
            BCE loss when ``use_sigmoid`` is True. Default None.
    """

    def __init__(self,
                 use_sigmoid=False,
                 use_soft=False,
                 use_balanced=False,
                 reduction='mean',
                 loss_weight=1.0,
                 class_weight=None,
                 pos_weight=None):
        super(CrossEntropyLoss, self).__init__()
        self.use_sigmoid = use_sigmoid
        self.use_soft = use_soft
        self.use_balanced=use_balanced
        assert not (
            self.use_soft and self.use_sigmoid
        ), 'use_sigmoid and use_soft could not be set simultaneously'

        self.reduction = reduction
        self.loss_weight = loss_weight
        self.class_weight = class_weight
        self.pos_weight = pos_weight

        if self.use_sigmoid:
            self.cls_criterion = binary_cross_entropy
        elif self.use_soft:
            self.cls_criterion = soft_cross_entropy
        elif self.use_balanced:
            self.cls_criterion=balanced_softmax_loss
        else:
            self.cls_criterion = cross_entropy

    def forward(self,
                cls_score,
                label,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)

        if self.class_weight is not None:
            class_weight = cls_score.new_tensor(self.class_weight)
        else:
            class_weight = None

        # only BCE loss has pos_weight
        if self.pos_weight is not None and self.use_sigmoid:
            pos_weight = cls_score.new_tensor(self.pos_weight)
            kwargs.update({'pos_weight': pos_weight})
        else:
            pos_weight = None

        loss_cls = self.loss_weight * self.cls_criterion(
            cls_score,
            label,
            weight,
            class_weight=class_weight,
            reduction=reduction,
            avg_factor=avg_factor,
            **kwargs)
        return loss_cls
