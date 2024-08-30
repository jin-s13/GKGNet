# Copyright (c) OpenMMLab. All rights reserved.
from .class_num_check_hook import ClassNumCheckHook
from .lr_updater import CosineAnnealingCooldownLrUpdaterHook
from .my_ema import MyEMAHook
    # OneCycleLrUpdaterHook
from .precise_bn_hook import PreciseBNHook

__all__ = [
    'ClassNumCheckHook', 'PreciseBNHook',
    'CosineAnnealingCooldownLrUpdaterHook','MyEMAHook'
    # 'OneCycleLrUpdaterHook'
]
