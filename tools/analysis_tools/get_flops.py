# Copyright (c) OpenMMLab. All rights reserved.
import argparse

from mmcv import Config
from mmcv.cnn.utils import get_model_complexity_info

from mmcls.models import build_classifier


def parse_args():
    parser = argparse.ArgumentParser(description='Get model flops and params')
    parser.add_argument('config', help='config file path')
    parser.add_argument(
        '--shape',
        type=int,
        nargs='+',
        default=[224, 224],
        help='input image size')
    args = parser.parse_args()
    return args


def main():

    args = parse_args()

    if len(args.shape) == 1:
        input_shape = (3, args.shape[0], args.shape[0])
    elif len(args.shape) == 2:
        input_shape = (3, ) + tuple(args.shape)
    else:
        raise ValueError('invalid input shape')

    cfg = Config.fromfile(args.config)
    model = build_classifier(cfg.model)
    model.eval()

    if hasattr(model, 'extract_feat'):
        model.forward = model.extract_feat
    else:
        raise NotImplementedError(
            'FLOPs counter is currently not currently supported with {}'.
            format(model.__class__.__name__))
    model = model.cuda()
    from torchprofile import profile_macs
    import torch
    shape=args.shape[0]
    flops = profile_macs(model , (torch.randn(1, 3, shape, shape)).cuda())
    print('flops: ', flops / 10 ** 9)
    print('para', sum([m.numel() for m in model.parameters()  if m.requires_grad]) / 10 ** 6)
    # flops, params = get_model_complexity_info(model, input_shape)
    # split_line = '=' * 30
    # print(f'{split_line}\nInput shape: {input_shape}\n'
    #       f'Flops: {flops}\nParams: {params}\n{split_line}')
    # print('!!!Please be cautious if you use the results in papers. '
    #       'You may need to check if all ops are supported and verify that the '
    #       'flops computation is correct.')


if __name__ == '__main__':
    main()
