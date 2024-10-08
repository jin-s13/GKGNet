# Copyright (c) OpenMMLab. All rights reserved.
from .augments import Augments
from .cutmix import BatchCutMixLayer
from .identity import Identity
from .mixup import BatchMixupLayer,BatchGenderMixupLayer

__all__ = ('Augments', 'BatchCutMixLayer', 'Identity', 'BatchMixupLayer','BatchGenderMixupLayer')
