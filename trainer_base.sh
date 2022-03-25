#!/bin/bash

bash tools/dist_train.sh configs/oln_box/baselines/hcoco_class_agn_faster_rcnn.py 4
bash tools/dist_test_bbox_evalclass.sh configs/oln_box/baselines/hcoco_class_agn_faster_rcnn.py out/oln_box/baselines/hcoco_class_agn_faster_rcnn/latest.pth 'nonhcoco' 4
bash tools/dist_test_bbox_evalclass.sh configs/oln_box/baselines/hcoco_class_agn_faster_rcnn.py out/oln_box/baselines/hcoco_class_agn_faster_rcnn/latest.pth 'hcoco' 4
bash tools/dist_test_bbox_evalclass.sh configs/oln_box/baselines/hcoco_class_agn_faster_rcnn.py out/oln_box/baselines/hcoco_class_agn_faster_rcnn/latest.pth 'all' 4

