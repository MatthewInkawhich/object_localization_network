#!/bin/bash

CUDA_VISIBLE_DEVICES=1,2,3 bash tools/dist_train.sh configs/oln_box/round0/animal_cz_hybrid_lc90_2x_r0.py 3
CUDA_VISIBLE_DEVICES=1,2,3 bash tools/dist_test_bbox_evalclass.sh configs/oln_box/round0/animal_cz_hybrid_lc90_2x_r0.py out/oln_box/round0/animal_cz_hybrid_lc90_2x_r0/latest.pth 'nonanimal' 3
CUDA_VISIBLE_DEVICES=1,2,3 bash tools/dist_test_bbox_evalclass.sh configs/oln_box/round0/animal_cz_hybrid_lc90_2x_r0.py out/oln_box/round0/animal_cz_hybrid_lc90_2x_r0/latest.pth 'animal' 3
CUDA_VISIBLE_DEVICES=1,2,3 bash tools/dist_test_bbox_evalclass.sh configs/oln_box/round0/animal_cz_hybrid_lc90_2x_r0.py out/oln_box/round0/animal_cz_hybrid_lc90_2x_r0/latest.pth 'all' 3
