_base_ = [
    '../_base_/datasets/imagenet_bs64_swin_224.py',
    '../_base_/default_runtime.py'
]
work_dir = './work_dirs/gkgnet_coco_576'
dataset_type = 'COCO'
# model settings
model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='GKGNet',
        choice='s',
        k=9,
        k_label_gcn=9,
        drop_path=0.1,
        n_classes=80,
        out_indices=(3,),
        size=576,
        init_cfg=dict(
            type='Pretrained',
            checkpoint='./checkpoint/pvig_s_82.1.pth.tar',
            prefix=None,
            map_location='cpu'
        ),
    ),
    neck=None,
    head=dict(
        type='LabelQueryHead',
        num_classes=80,
        in_channels=640,
        softmax=  False,
        loss=dict(  type='AsymmetricLoss',
                    gamma_pos=0.0,
                    gamma_neg=2.0,
                    clip=0.05,
                    ),
        topk=(1, 1),
    ),
    )
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
scale_size=600
crop_size=576

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='CropMixup', p=0.5, size=crop_size, scale=0.01, number=234),
    dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
    dict(
        type='RandomErasing',
        erase_prob=0.5,
        mode='rand',
        min_area_ratio=0.02,
        max_area_ratio=1 / 3,
        fill_color=img_norm_cfg['mean'][::-1],
        fill_std=img_norm_cfg['std'][::-1]),
    dict(type='ColorJitter', brightness=0.4, contrast=0.4, saturation=0.4),
    dict(type='Trivial', p=1.0),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=['gt_label']),
    dict(type='Collect', keys=['img',
                               'gt_label',
                               ])
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='Resize',
        size=crop_size,
        backend='pillow',
        interpolation='bicubic'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='Collect', keys=['img',
                               # 'gt_label'
                               ])
]
sampler = dict(type='RepeatAugSampler')
data = dict(
    samples_per_gpu=16,
    workers_per_gpu=1,
    train=
    dict(
        type='ClassBalancedDataset',
        oversample_thr=0.01,
        dataset=dict(
            type=dataset_type,
            data_prefix='../0data/coco/train2014',
            ann_file='../0data/coco/train.data',
            pipeline=train_pipeline),
    ),
    val=dict(
        type=dataset_type,
        data_prefix='../0data/coco/val2014',
        ann_file='../0data/coco/val_test.data',
        pipeline=test_pipeline,
        test_mode=True),
    test=
    dict(
        type=dataset_type,
        data_prefix='../0data/coco/val2014',
        ann_file='../0data/coco/val_test.data',
        pipeline=test_pipeline,
        test_mode=True),
)
evaluation = dict(interval=1, metric='accuracy',save_best='mAP')
runner = dict(type='EpochBasedRunner', max_epochs=80)
paramwise_cfg = dict(
    norm_decay_mult=0.0,
    bias_decay_mult=0.0,
    custom_keys={
        '.absolute_pos_embed': dict(decay_mult=0.0),
        '.relative_position_bias_table': dict(decay_mult=0.0),
    }
)

optimizer = dict(
    type='AdamW',
    lr=1e-4,
    weight_decay=0.05,
    eps=1e-08,
    betas=(0.9, 0.999),
    paramwise_cfg=paramwise_cfg)
optimizer_config = dict(grad_clip=dict(max_norm=5.0))
lr_config = dict(
    policy='step',
    step=[10,50],
    warmup='linear',
    warmup_ratio=1e-3,
    warmup_iters=5,
    warmup_by_epoch=True
)
# # yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])
dist_params = dict(backend='nccl')
log_level = 'INFO'
workflow = [('train', 1)]
# 表示动态 scale
fp16 = dict(loss_scale='dynamic')