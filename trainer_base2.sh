#!/bin/bash

#bash tools/dist_train.sh configs/oln_box/baselines/voc_ssl_p05_s1_rpn.py 4
bash tools/dist_test_bbox_evalclass.sh configs/oln_box/baselines/voc_ssl_p05_s1_rpn.py out/oln_box/baselines/voc_ssl_p05_s1_rpn/latest.pth 'nonvoc' 4
bash tools/dist_test_bbox_evalclass.sh configs/oln_box/baselines/voc_ssl_p05_s1_rpn.py out/oln_box/baselines/voc_ssl_p05_s1_rpn/latest.pth 'voc' 4
bash tools/dist_test_bbox_evalclass.sh configs/oln_box/baselines/voc_ssl_p05_s1_rpn.py out/oln_box/baselines/voc_ssl_p05_s1_rpn/latest.pth 'all' 4

#bash tools/dist_train.sh configs/oln_box/baselines/voc_ssl_p10_s1_rpn.py 4
bash tools/dist_test_bbox_evalclass.sh configs/oln_box/baselines/voc_ssl_p10_s1_rpn.py out/oln_box/baselines/voc_ssl_p10_s1_rpn/latest.pth 'nonvoc' 4
bash tools/dist_test_bbox_evalclass.sh configs/oln_box/baselines/voc_ssl_p10_s1_rpn.py out/oln_box/baselines/voc_ssl_p10_s1_rpn/latest.pth 'voc' 4
bash tools/dist_test_bbox_evalclass.sh configs/oln_box/baselines/voc_ssl_p10_s1_rpn.py out/oln_box/baselines/voc_ssl_p10_s1_rpn/latest.pth 'all' 4

#bash tools/dist_train.sh configs/oln_box/baselines/voc_ssl_p25_s1_rpn.py 4
bash tools/dist_test_bbox_evalclass.sh configs/oln_box/baselines/voc_ssl_p25_s1_rpn.py out/oln_box/baselines/voc_ssl_p25_s1_rpn/latest.pth 'nonvoc' 4
bash tools/dist_test_bbox_evalclass.sh configs/oln_box/baselines/voc_ssl_p25_s1_rpn.py out/oln_box/baselines/voc_ssl_p25_s1_rpn/latest.pth 'voc' 4
bash tools/dist_test_bbox_evalclass.sh configs/oln_box/baselines/voc_ssl_p25_s1_rpn.py out/oln_box/baselines/voc_ssl_p25_s1_rpn/latest.pth 'all' 4

#bash tools/dist_train.sh configs/oln_box/baselines/voc_ssl_p50_s1_rpn.py 4
bash tools/dist_test_bbox_evalclass.sh configs/oln_box/baselines/voc_ssl_p50_s1_rpn.py out/oln_box/baselines/voc_ssl_p50_s1_rpn/latest.pth 'nonvoc' 4
bash tools/dist_test_bbox_evalclass.sh configs/oln_box/baselines/voc_ssl_p50_s1_rpn.py out/oln_box/baselines/voc_ssl_p50_s1_rpn/latest.pth 'voc' 4
bash tools/dist_test_bbox_evalclass.sh configs/oln_box/baselines/voc_ssl_p50_s1_rpn.py out/oln_box/baselines/voc_ssl_p50_s1_rpn/latest.pth 'all' 4

