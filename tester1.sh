#!/bin/bash

bash tools/dist_test_bbox.sh configs/oln_box/baselines/voc_class_agn_faster_rcnn.py out/oln_box/baselines/voc_class_agn_faster_rcnn/latest.pth 4
echo "HERE"
bash tools/dist_test_bbox.sh configs/oln_box/baselines/voc5_class_agn_faster_rcnn.py out/oln_box/baselines/voc5_class_agn_faster_rcnn/latest.pth 4
echo "HERE"
bash tools/dist_test_bbox.sh configs/oln_box/baselines/animal_class_agn_faster_rcnn.py out/oln_box/baselines/animal_class_agn_faster_rcnn/latest.pth 4
echo "HERE"
bash tools/dist_test_bbox.sh configs/oln_box/baselines/vehicle_class_agn_faster_rcnn.py out/oln_box/baselines/vehicle_class_agn_faster_rcnn/latest.pth 4
echo "HERE"

