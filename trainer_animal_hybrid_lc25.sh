#!/bin/bash

bash tools/dist_train.sh configs/oln_box/round0/animal_cz_hybrid_lc25_2x_r0.py 4

bash tools/dist_test_bbox_evalclass.sh configs/oln_box/round0/animal_cz_hybrid_lc25_2x_r0.py out/oln_box/round0/animal_cz_hybrid_lc25_2x_r0/latest.pth 'nonanimal' 4
bash tools/dist_test_bbox_evalclass.sh configs/oln_box/round0/animal_cz_hybrid_lc25_2x_r0.py out/oln_box/round0/animal_cz_hybrid_lc25_2x_r0/latest.pth 'animal' 4
bash tools/dist_test_bbox_evalclass.sh configs/oln_box/round0/animal_cz_hybrid_lc25_2x_r0.py out/oln_box/round0/animal_cz_hybrid_lc25_2x_r0/latest.pth 'all' 4
