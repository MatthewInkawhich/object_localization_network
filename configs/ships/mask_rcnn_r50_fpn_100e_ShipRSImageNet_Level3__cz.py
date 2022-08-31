_base_ = [
    '../_base_/models/mask_rcnn_r50_fpn.py',
    '../_base_/datasets/ShipRSImageNet_Level3_instance.py',
    '../_base_/schedules/schedule_100e.py', '../_base_/default_runtime.py',
]
model = dict(
    roi_head=dict(
        bbox_head=dict(
            num_classes=50),
        mask_head=dict(
            num_classes=50)))



dataset_type = 'ShipRSImageNet_Level3'
data_root = './data/ShipRSImageNet/'

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    #dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
    dict(type='MinIoURandomCrop', min_ious=(0.1, 0.3, 0.5, 0.7, 0.9), min_crop_size=0.3),  # crop
    dict(type='Resize', img_scale=(930, 930), keep_ratio=False),  # zoom
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(930, 930),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'COCO_Format/ShipRSImageNet_bbox_train_level_3.json',
        img_prefix=data_root + 'VOC_Format/JPEGImages/',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'COCO_Format/ShipRSImageNet_bbox_val_level_3.json',
        img_prefix=data_root + 'VOC_Format/JPEGImages/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'COCO_Format/ShipRSImageNet_bbox_val_level_3.json',
        img_prefix=data_root + 'VOC_Format/JPEGImages/',
        pipeline=test_pipeline))



checkpoint_config = dict(interval=25)
evaluation = dict(interval=50, metric=['bbox', 'segm'])

work_dir='./out/ships/mask_rcnn_r50_fpn_100e_ShipRSImageNet_Level3__cz'
