#!/bin/bash


### Round 0
bash tools/dist_train.sh configs/oln_box/round0/animal_cz_2x_r0.py 4
bash tools/dist_test_bbox.sh configs/oln_box/round0/animal_cz_2x_r0.py out/oln_box/round0/animal_cz_2x_r0/latest.pth 4

bash tools/dist_train.sh configs/oln_box/round0/vehicle_cz_2x_r0.py 4
bash tools/dist_test_bbox.sh configs/oln_box/round0/vehicle_cz_2x_r0.py out/oln_box/round0/vehicle_cz_2x_r0/latest.pth 4
