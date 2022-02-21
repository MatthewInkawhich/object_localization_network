#!/bin/bash

#bash tools/dist_test_bbox.sh configs/oln_box/baselines/voc_oln.py out/oln_box/baselines/voc_oln/latest.pth 4
#echo "HERE"
#bash tools/dist_test_bbox.sh configs/oln_box/baselines/voc5_oln.py out/oln_box/baselines/voc5_oln/latest.pth 4
#echo "HERE"
bash tools/dist_test_bbox.sh configs/oln_box/baselines/animal_oln.py out/oln_box/baselines/animal_oln/latest.pth 4
echo "HERE"
bash tools/dist_test_bbox.sh configs/oln_box/baselines/vehicle_oln.py out/oln_box/baselines/vehicle_oln/latest.pth 4
echo "HERE"

