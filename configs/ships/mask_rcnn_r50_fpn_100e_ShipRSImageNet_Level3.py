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


checkpoint_config = dict(interval=20)
evaluation = dict(interval=50, metric=['bbox', 'segm'])

work_dir='./out/ships/mask_rcnn_r50_fpn_100e_ShipRSImageNet_Level3'
