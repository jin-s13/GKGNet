# Copyright (c) OpenMMLab. All rights reserved.
from .gap import GlobalAveragePooling
from .hr_fuse import HRFuseScales
from .multi_label_projection import MultiLabelProjection
# from .channel_mapper import ChannelMapper

__all__ = ['GlobalAveragePooling', 'HRFuseScales',
           'MultiLabelProjection',
           # 'ChannelMapper'
           ]
