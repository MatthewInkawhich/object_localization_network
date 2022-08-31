#!/bin/bash

bash tools/dist_train.sh configs/ships/mask_rcnn_r50_fpn_100e_ShipRSImageNet_Level2.py 4
bash tools/dist_test_bbox.sh configs/ships/mask_rcnn_r50_fpn_100e_ShipRSImageNet_Level2.py out/ships/mask_rcnn_r50_fpn_100e_ShipRSImageNet_Level2/latest.pth 4

