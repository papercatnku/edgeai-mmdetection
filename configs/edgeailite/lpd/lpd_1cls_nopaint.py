_base_ = [
    '../_xbase_/hyper_params/yolox_config.py',
    '../_xbase_/hyper_params/yolox_schedule.py',
]

img_scale = (640,640)
input_size = img_scale
samples_per_gpu = 32

CLASSES =["LP"]
    

dataset_type = 'CocoDataset'
num_classes = len(CLASSES)
img_prefix = '/media/112new_sde/LPD/DTC_RAW/'
data_root = '/media/112new_sde/LPD/LPDAnnotations/coco_style/cls_1_nopaint/'
img_norm_cfg = dict(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0], to_rgb=False)

# quantization 
quantitize = None # 
convert_to_lite_model = dict(group_size_dw=None)

interval = 10



# no quantitize setting
if quantitize:
    samples_per_gpu = samples_per_gpu//2
    load_from = './work_dirs/yolox_s_lite/latest.pth'
    max_epochs = (1 if quantitize == 'calibration' else 12)
    initial_learning_rate = 1e-4
    num_last_epochs = max_epochs//2
    interval = 1
    resume_from = None
else:
    load_from = None 
    max_epochs = 100
    initial_learning_rate = 0.01
    num_last_epochs = 10
    interval = 5
    resume_from = None

use_depth_wise=False
test_backbone_cfg = dict(
    type='CSPDarknet',
    arch='P4',
    deepen_factor=0.5,
    widen_factor=0.5,
    out_indices=(3,),    
    use_depthwise=use_depth_wise,
    spp_kernal_sizes=(5, 9, 13),
    act_cfg=dict(type='LeakyReLU', negative_slope=0.1)
    )
test_neck_cfg = dict(
    type='YOLOXPAFPN',
    in_channels=[256,],
    out_channels=256,
    num_csp_blocks=2,
    use_depthwise=use_depth_wise,
    upsample_cfg=dict(scale_factor=2, mode='bilinear'),
    conv_cfg=None,
    norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
    act_cfg=dict(type='LeakyReLU', negative_slope=0.1),
    )
test_head_cfg = dict(
    type='YOLOXHead',
    num_classes=num_classes,
    in_channels=256,
    feat_channels=128,
    strides=[16,],
    use_depthwise=True,
    act_cfg=dict(type='LeakyReLU', negative_slope=0.1)             
    )

model = dict(
    type='YOLOX',
    input_size=input_size,
    random_size_range=(10, 20),
    random_size_interval=10,
    backbone=test_backbone_cfg,
    neck=test_neck_cfg,
    bbox_head=test_head_cfg,
    train_cfg = dict(
    assigner=dict(type='SimOTAAssigner', center_radius=2.5)),
    test_cfg=dict(score_thr=0.01, nms=dict(type='nms', iou_threshold=0.65))
)


train_pipeline = [
    dict(type='Mosaic', img_scale=img_scale, pad_val=114.0),
    dict(
        type='RandomAffine',
        scaling_ratio_range=(0.5, 1.5),
        border=(-img_scale[0] // 2, -img_scale[1] // 2)),
    dict(
        type='MixUp',
        img_scale=img_scale,
        ratio_range=(0.8, 1.6),
        pad_val=114.0),
    dict(type='YOLOXHSVRandomAug'),
    dict(type='RandomFlip', flip_ratio=0.5),
    # According to the official implementation, multi-scale
    # training is not considered here but in the
    # 'mmdet/models/detectors/yolox.py'.
    dict(type='Resize', img_scale=img_scale, keep_ratio=True),
    dict(
        type='Pad',
        pad_to_square=True,
        # If the image is three-channel, the pad value needs
        # to be set separately for each channel.
        pad_val=dict(img=(114.0, 114.0, 114.0))),
    dict(type='FilterAnnotations', min_gt_bbox_wh=(1, 1), keep_empty=False),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=img_scale,
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(
                type='Pad',
                pad_to_square=True,
                pad_val=dict(img=(114.0, 114.0, 114.0))),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img'])
        ])
]


train_dataset = dict(
    type='MultiImageMixDataset',
    dataset=dict(
        type=dataset_type,
        ann_file=data_root + 'train_annotations.json',
        classes=CLASSES,
        img_prefix=img_prefix,
        filter_empty_gt=False,
        pipeline=[
                dict(type='LoadImageFromFile'),
                dict(type='LoadAnnotations', with_bbox=True)
            ]
    ),
        pipeline=train_pipeline),
    

data = dict(
    samples_per_gpu=samples_per_gpu,
    workers_per_gpu=2,
    persistent_workers=True,
    train=train_dataset,
        
    val=dict(type=dataset_type,
        ann_file=data_root + 'val_annotations.json',
        classes=CLASSES,
        img_prefix=img_prefix,
        pipeline=test_pipeline),

    test=dict(type=dataset_type,
        ann_file=data_root + 'val_annotations.json',
        classes=CLASSES,
        img_prefix=img_prefix,
        pipeline=test_pipeline))

runner = dict(type='EpochBasedRunner', max_epochs=max_epochs)

custom_hooks = [
    dict(
        type='YOLOXModeSwitchHook',
        num_last_epochs=num_last_epochs,
        priority=48),
    dict(
        type='SyncNormHook',
        num_last_epochs=num_last_epochs,
        interval=interval,
        priority=48),
    dict(
        type='ExpMomentumEMAHook',
        resume_from=resume_from,
        momentum=0.0001,
        priority=49)
]

# optimizer = dict(
#     type='SGD',lr=initial_learning_rate, momentum=0.9,
#     weight_decay=5e-4,
#     nesterov=True,
#     paramwise_cfg=dict(norm_decay_mult=0., bias_decay_mult=0.)
# )

# optimizer_config = dict(grad_clip=None,detect_anomalous_params=True)


# lr_config = dict(
#     num_last_epochs=num_last_epochs,
#     policy='YOLOX',
#     warmup='exp',
#     by_epoch=False,
#     warmup_by_epoch=True,
#     warmup_ratio=1,
#     warmup_iters=5,  # 5 epoch
#     num_last_epochs=num_last_epochs,
#     min_lr_ratio=0.05)

evaluation = dict(
    save_best='auto',
    # The evaluation interval is 'interval' when running epoch is
    # less than ‘max_epochs - num_last_epochs’.
    # The evaluation interval is 1 when running epoch is greater than
    # or equal to ‘max_epochs - num_last_epochs’.
    interval=interval,
    dynamic_intervals=[(max_epochs - num_last_epochs, 1)],
    metric='bbox',
    classwise=True,
    )

log_config = dict(interval=50)

checkpoint_config = dict(interval=interval)




