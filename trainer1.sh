#!/bin/bash

#bash tools/dist_train.sh configs/oln_box/baselines/voc5_oln_2x.py 4
#bash tools/dist_test_bbox.sh configs/oln_box/baselines/voc5_oln_2x.py out/oln_box/baselines/voc5_oln_2x/latest.pth 4

#bash tools/dist_train.sh configs/oln_box/baselines/animal_oln_2x.py 4
#bash tools/dist_test_bbox.sh configs/oln_box/baselines/animal_oln_2x.py out/oln_box/baselines/animal_oln_2x/latest.pth 4

#bash tools/dist_train.sh configs/oln_box/baselines/vehicle_oln_2x.py 4
#bash tools/dist_test_bbox.sh configs/oln_box/baselines/vehicle_oln_2x.py out/oln_box/baselines/vehicle_oln_2x/latest.pth 4

#bash tools/dist_train_mixup.sh configs/oln_box/round0/animal_mixcz_qfl_r0.py 4
#bash tools/dist_test_bbox.sh configs/oln_box/round0/animal_mixcz_qfl_r0.py out/oln_box/round0/animal_mixcz_qfl_r0/latest.pth 4

#bash tools/dist_train_mixup.sh configs/oln_box/round0/vehicle_mixcz_qfl_r0.py 4
#bash tools/dist_test_bbox.sh configs/oln_box/round0/vehicle_mixcz_qfl_r0.py out/oln_box/round0/vehicle_mixcz_qfl_r0/latest.pth 4


bash tools/dist_train.sh configs/oln_box/round1/voc_cz_qfl_2x_r1_p35.py 4
bash tools/dist_test_bbox.sh configs/oln_box/round1/voc_cz_qfl_2x_r1_p35.py out/oln_box/round1/voc_cz_qfl_2x_r1_p35/latest.pth 4
