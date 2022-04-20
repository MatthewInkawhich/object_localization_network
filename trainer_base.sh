#!/bin/bash

#bash tools/dist_train.sh configs/oln_box/baselines/voc_rpn.py 4
bash tools/dist_test_bbox_evalclass.sh configs/oln_box/baselines/voc_rpn.py out/oln_box/baselines/voc_rpn/latest.pth 'nonvoc' 4
bash tools/dist_test_bbox_evalclass.sh configs/oln_box/baselines/voc_rpn.py out/oln_box/baselines/voc_rpn/latest.pth 'voc' 4
bash tools/dist_test_bbox_evalclass.sh configs/oln_box/baselines/voc_rpn.py out/oln_box/baselines/voc_rpn/latest.pth 'all' 4

#bash tools/dist_train.sh configs/oln_box/baselines/voc5_rpn.py 4
bash tools/dist_test_bbox_evalclass.sh configs/oln_box/baselines/voc5_rpn.py out/oln_box/baselines/voc5_rpn/latest.pth 'nonvoc5' 4
bash tools/dist_test_bbox_evalclass.sh configs/oln_box/baselines/voc5_rpn.py out/oln_box/baselines/voc5_rpn/latest.pth 'voc5' 4
bash tools/dist_test_bbox_evalclass.sh configs/oln_box/baselines/voc5_rpn.py out/oln_box/baselines/voc5_rpn/latest.pth 'all' 4

#bash tools/dist_train.sh configs/oln_box/baselines/animal_rpn.py 4
bash tools/dist_test_bbox_evalclass.sh configs/oln_box/baselines/animal_rpn.py out/oln_box/baselines/animal_rpn/latest.pth 'nonanimal' 4
bash tools/dist_test_bbox_evalclass.sh configs/oln_box/baselines/animal_rpn.py out/oln_box/baselines/animal_rpn/latest.pth 'animal' 4
bash tools/dist_test_bbox_evalclass.sh configs/oln_box/baselines/animal_rpn.py out/oln_box/baselines/animal_rpn/latest.pth 'all' 4

#bash tools/dist_train.sh configs/oln_box/baselines/hcoco_rpn.py 4
bash tools/dist_test_bbox_evalclass.sh configs/oln_box/baselines/hcoco_rpn.py out/oln_box/baselines/hcoco_rpn/latest.pth 'nonhcoco' 4
bash tools/dist_test_bbox_evalclass.sh configs/oln_box/baselines/hcoco_rpn.py out/oln_box/baselines/hcoco_rpn/latest.pth 'hcoco' 4
bash tools/dist_test_bbox_evalclass.sh configs/oln_box/baselines/hcoco_rpn.py out/oln_box/baselines/hcoco_rpn/latest.pth 'all' 4
