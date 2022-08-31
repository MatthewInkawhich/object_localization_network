#!/bin/bash

bash tools/dist_train.sh configs/ships/mask_rcnn_r50_fpn_100e_ShipRSImageNet_Level3__cz.py 4
bash tools/dist_test_bbox.sh configs/ships/mask_rcnn_r50_fpn_100e_ShipRSImageNet_Level3__cz.py out/ships/mask_rcnn_r50_fpn_100e_ShipRSImageNet_Level3__cz/latest.pth 4

