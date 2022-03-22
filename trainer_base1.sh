#!/bin/bash

### OLN baselines
bash tools/dist_train.sh configs/oln_box/baselines/animal_ssl_p05_s1_class_agn_faster_rcnn.py 4
bash tools/dist_test_bbox.sh configs/oln_box/baselines/animal_ssl_p05_s1_class_agn_faster_rcnn.py out/oln_box/baselines/animal_ssl_p05_s1_class_agn_faster_rcnn/latest.pth 4
echo "DONE: SSL P05"

bash tools/dist_train.sh configs/oln_box/baselines/animal_ssl_p10_s1_class_agn_faster_rcnn.py 4
bash tools/dist_test_bbox.sh configs/oln_box/baselines/animal_ssl_p10_s1_class_agn_faster_rcnn.py out/oln_box/baselines/animal_ssl_p10_s1_class_agn_faster_rcnn/latest.pth 4
echo "DONE: SSL P10"

bash tools/dist_train.sh configs/oln_box/baselines/animal_ssl_p25_s1_class_agn_faster_rcnn.py 4
bash tools/dist_test_bbox.sh configs/oln_box/baselines/animal_ssl_p25_s1_class_agn_faster_rcnn.py out/oln_box/baselines/animal_ssl_p25_s1_class_agn_faster_rcnn/latest.pth 4
echo "DONE: SSL P25"

bash tools/dist_train.sh configs/oln_box/baselines/animal_ssl_p50_s1_class_agn_faster_rcnn.py 4
bash tools/dist_test_bbox.sh configs/oln_box/baselines/animal_ssl_p50_s1_class_agn_faster_rcnn.py out/oln_box/baselines/animal_ssl_p50_s1_class_agn_faster_rcnn/latest.pth 4
echo "DONE: SSL P50"
