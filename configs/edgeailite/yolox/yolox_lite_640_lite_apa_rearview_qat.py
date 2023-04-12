# modified from: https://github.com/open-mmlab/mmdetection/tree/master/configs/yolox

_base_ = [
    # f'../_xbase_/datasets/{dataset_type.lower()}.py',
    '../_xbase_/hyper_params/yolox_config.py',
    '../_xbase_/hyper_params/yolox_schedule.py',
]

img_scale = (480, 640)
input_size = img_scale
samples_per_gpu = 64


CLASSES = (
        "road_cone", "car_barrier", "ground_lock", "road_pile", 
        "warning_board", "shopping_cart", "ball_stone", "sharing_bike",
        "electric_bike", "child", "others",
        )
CLASSES_MAPS = {
    }
CLASSES_IGNORES = []

# dataset settings
dataset_type = 'APAObstacleDataset'
num_classes = len(CLASSES)
data_root = "/media/104sdf/tb9919/Datasets/APAObstacleDetection/rearview_obstacle_det"
# work_dir="./work_dirs/yolox_nano_lite_apa_rearview_v001qat"

img_norm_cfg = dict(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0], to_rgb=False)

# replace complex activation functions with ReLU.
# Also replace regular convolutions with depthwise-separable convolutions.
# torchvision.edgeailite requires edgeai-torchvision to be installed
convert_to_lite_model = dict(group_size_dw=None)

# settings for qat or calibration - set to True after doing floating point training
# quantize = False #'training' #'calibration'
quantize = True #'training' #'calibration'
if quantize:
    samples_per_gpu = samples_per_gpu//2
    load_from = '/media/112new_sde/ModelZoo/lpd/apa_rearview_lite/epoch_300.pth'
    max_epochs = (1 if quantize == 'calibration' else 30)
    initial_learning_rate = 1e-4
    num_last_epochs = max_epochs//2
    interval = 2
    resume_from = None
else:
    load_from = None #'https://download.openmmlab.com/mmdetection/v2.0/yolox/yolox_s_8x8_300e_coco/yolox_s_8x8_300e_coco_20211121_095711-4592a793.pth'
    max_epochs = 300
    initial_learning_rate = 0.01
    num_last_epochs = 20
    interval = 5
    resume_from = None
#

# model settings
use_depthwise=False
model = dict(
    type='YOLOX',
    input_size=img_scale,
    random_size_range=(14, 16),
    random_size_interval=10,
    backbone=dict(
        arch='P4',
        type='CSPDarknet', 
        deepen_factor=0.333, 
        widen_factor=0.25, 
        out_indices=(3,), 
        use_depthwise=use_depthwise,
        act_cfg=dict(type='LeakyReLU', negative_slope=0.1)
        ),
    neck=dict(
        type='YOLOXPAFPN',
        in_channels=[128,],
        out_channels=128,
        num_csp_blocks=1,
        use_depthwise=use_depthwise,
        act_cfg=dict(type='LeakyReLU', negative_slope=0.1),
        ),
    bbox_head=dict(
        type='YOLOXHead',
        num_classes=num_classes, 
        in_channels=128, 
        feat_channels=128, 
        use_depthwise=use_depthwise,
        strides=[16,],
        act_cfg=dict(type='LeakyReLU', negative_slope=0.1)
        ),
    train_cfg=dict(assigner=dict(type='SimOTAAssigner', center_radius=2.5)),
    # In order to align the source code, the threshold of the val phase is
    # 0.01, and the threshold of the test phase is 0.001.
    test_cfg=dict(score_thr=0.01, nms=dict(type='nms', iou_threshold=0.65))
    )

train_pipeline = [
    # dict(type='Mosaic', img_scale=img_scale, pad_val=114.0),
    # dict(
    #     type='RandomAffine',
    #     scaling_ratio_range=(0.5, 1.5),
    #     border=(-img_scale[0] // 2, -img_scale[1] // 2)),
    # dict(
    #     type='MixUp',
    #     img_scale=img_scale,
    #     ratio_range=(0.8, 1.6),
    #     pad_val=114.0),
    dict(type='YOLOXHSVRandomAug'),
    dict(type='RandomFlip', flip_ratio=0.5),
    # According to the official implementation, multi-scale
    # training is not considered here but in the
    # 'mmdet/models/detectors/yolox.py'.
    dict(type='Resize', img_scale=img_scale, keep_ratio=True),
    dict(
        type='Pad',
        size=input_size,
        pad_to_square=False,
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
                size=input_size,
                pad_to_square=False,
                pad_val=dict(img=(114.0, 114.0, 114.0))
                ),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img'])
        ])
]

data = dict(
    samples_per_gpu=samples_per_gpu,
    workers_per_gpu=2,
    persistent_workers=True,
    train=dict(
        type='MultiImageMixDataset',
        dataset=dict(
            type=dataset_type,
            ann_file=data_root + "/train.list",
            data_root=data_root,
            classes=CLASSES,
            classes_maps=CLASSES_MAPS,
            classes_ignores=CLASSES_IGNORES,
            filter_empty_gt=False,
            pipeline=[
                dict(type='LoadImageFromFile'),
                dict(type='LoadAnnotations', with_bbox=True)
            ]
        ),
        pipeline=train_pipeline
    ),
    val=dict(
        type=dataset_type,
        ann_file=data_root + '/val.list',
        data_root=data_root,
        classes=CLASSES,
        classes_maps=CLASSES_MAPS,
        classes_ignores=CLASSES_IGNORES,
        pipeline=test_pipeline
    ),
    test=dict(
        type=dataset_type,
        ann_file=data_root + '/val.list',
        data_root=data_root,
        classes=CLASSES,
        classes_maps=CLASSES_MAPS,
        classes_ignores=CLASSES_IGNORES,
        pipeline=test_pipeline
    ),
)

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
        priority=49),
]

optimizer = dict(type='SGD', lr=initial_learning_rate)
lr_config = dict(num_last_epochs=num_last_epochs)
# evaluation = dict(interval=5, metric='bbox')
evaluation = dict(interval=interval, metric='mAP')

checkpoint_config = dict(
    interval=interval)