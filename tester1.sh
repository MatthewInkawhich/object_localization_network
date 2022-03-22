#!/bin/bash

bash tools/dist_test_bbox_evalclass.sh configs/oln_box/baselines/animal_ssl_p05_s1_class_agn_faster_rcnn.py out/oln_box/baselines/animal_ssl_p05_s1_class_agn_faster_rcnn/latest.pth 'animal' 4
echo "FINISHED ID"
bash tools/dist_test_bbox_evalclass.sh configs/oln_box/baselines/animal_ssl_p05_s1_class_agn_faster_rcnn.py out/oln_box/baselines/animal_ssl_p05_s1_class_agn_faster_rcnn/latest.pth 'all' 4
echo "FINISHED ALL"

bash tools/dist_test_bbox_evalclass.sh configs/oln_box/baselines/animal_ssl_p05_s1_oln_2x.py out/oln_box/baselines/animal_ssl_p05_s1_oln_2x/latest.pth 'animal' 4
echo "FINISHED ID"
bash tools/dist_test_bbox_evalclass.sh configs/oln_box/baselines/animal_ssl_p05_s1_oln_2x.py out/oln_box/baselines/animal_ssl_p05_s1_oln_2x/latest.pth 'all' 4
echo "FINISHED ALL"

bash tools/dist_test_bbox_evalclass.sh configs/oln_box/round0/animal_ssl_p05_s1_cz_2x_r0.py out/oln_box/round0/animal_ssl_p05_s1_cz_2x_r0/latest.pth 'animal' 4
echo "FINISHED ID"
bash tools/dist_test_bbox_evalclass.sh configs/oln_box/round0/animal_ssl_p05_s1_cz_2x_r0.py out/oln_box/round0/animal_ssl_p05_s1_cz_2x_r0/latest.pth 'all' 4
echo "FINISHED ALL"


