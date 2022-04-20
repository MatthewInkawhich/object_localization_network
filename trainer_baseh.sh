#!/bin/bash

bash tools/dist_train.sh configs/oln_box/baselines/hanimal_class_agn_faster_rcnn.py 4
bash tools/dist_test_bbox_evalclass.sh configs/oln_box/baselines/hanimal_class_agn_faster_rcnn.py out/oln_box/baselines/hanimal_class_agn_faster_rcnn/latest.pth 'nonhanimal' 4
bash tools/dist_test_bbox_evalclass.sh configs/oln_box/baselines/hanimal_class_agn_faster_rcnn.py out/oln_box/baselines/hanimal_class_agn_faster_rcnn/latest.pth 'hanimal' 4
bash tools/dist_test_bbox_evalclass.sh configs/oln_box/baselines/hanimal_class_agn_faster_rcnn.py out/oln_box/baselines/hanimal_class_agn_faster_rcnn/latest.pth 'all' 4

bash tools/dist_train.sh configs/oln_box/baselines/hanimal_oln_2x.py 4
bash tools/dist_test_bbox_evalclass.sh configs/oln_box/baselines/hanimal_oln_2x.py out/oln_box/baselines/hanimal_oln_2x/latest.pth 'nonhanimal' 4
bash tools/dist_test_bbox_evalclass.sh configs/oln_box/baselines/hanimal_oln_2x.py out/oln_box/baselines/hanimal_oln_2x/latest.pth 'hanimal' 4
bash tools/dist_test_bbox_evalclass.sh configs/oln_box/baselines/hanimal_oln_2x.py out/oln_box/baselines/hanimal_oln_2x/latest.pth 'all' 4

bash tools/dist_train.sh configs/oln_box/round0/hanimal_cz_2x_r0.py 4
bash tools/dist_test_bbox_evalclass.sh configs/oln_box/round0/hanimal_cz_2x_r0.py out/oln_box/round0/hanimal_cz_2x_r0/latest.pth 'nonhanimal' 4
bash tools/dist_test_bbox_evalclass.sh configs/oln_box/round0/hanimal_cz_2x_r0.py out/oln_box/round0/hanimal_cz_2x_r0/latest.pth 'hanimal' 4
bash tools/dist_test_bbox_evalclass.sh configs/oln_box/round0/hanimal_cz_2x_r0.py out/oln_box/round0/hanimal_cz_2x_r0/latest.pth 'all' 4
