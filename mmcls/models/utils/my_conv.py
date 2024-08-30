import torch
import torch.nn as nn
from torch import Tensor

from mmcv.cnn.bricks.registry import CONV_LAYERS
from torch.nn.common_types import _size_1_t, _size_2_t, _size_3_t
from typing import Optional, List, Tuple, Union
class AcCon2d(nn.Conv2d):
    '''
    单独处理交流信号的conv2d
    '''
    def forward(self, input: Tensor) -> Tensor:
        # input:(B,C,H,W)
        # keep E(dc)=2E(ac)

        input_flat = input.flatten(2)
        # input_ac = input_flat - input_flat.mean(-1)[:, :, None]
        input_dc = input_flat.mean(-1)
        # e_ac = input_ac.norm(2, dim=1).mean()
        # e_dc = input_dc.norm(2, dim=1).mean()
        # e_ac2dc = e_ac / e_dc
        # alpha = 2 * e_ac2dc - 1
        alpha=-1.0
        input = input + alpha * input_dc[:, :, None, None]

        return self._conv_forward(input, self.weight, self.bias)


# CONV_LAYERS.register_module('Conv2d', force=True,module=AcCon2d)



def build_conv_layer(cfg, *args, **kwargs):
    """Build convolution layer.

    Args:
        cfg (None or dict): The conv layer config, which should contain:
            - type (str): Layer type.
            - layer args: Args needed to instantiate an conv layer.
        args (argument list): Arguments passed to the `__init__`
            method of the corresponding conv layer.
        kwargs (keyword arguments): Keyword arguments passed to the `__init__`
            method of the corresponding conv layer.

    Returns:
        nn.Module: Created conv layer.
    """
    if cfg is None:
        cfg_ = dict(type='Conv2d')
    else:
        if not isinstance(cfg, dict):
            raise TypeError('cfg must be a dict')
        if 'type' not in cfg:
            raise KeyError('the cfg dict must contain the key "type"')
        cfg_ = cfg.copy()

    layer_type = cfg_.pop('type')
    if layer_type not in CONV_LAYERS:
        raise KeyError(f'Unrecognized norm type {layer_type}')
    else:
        conv_layer = CONV_LAYERS.get(layer_type)

    layer = conv_layer(*args, **kwargs, **cfg_)

    return layer


