#!/bin/bash

bash tools/dist_train.sh configs/oln_box_ships/round0/allship_cz_hybrid_lc10_r0.py 4
bash tools/dist_test_bbox_evalclass.sh configs/oln_box_ships/round0/allship_cz_hybrid_lc10_r0.py out/oln_box_ships/round0/allship_cz_hybrid_lc10_r0/latest.pth 'allship' 4
echo "DONE LC10"

bash tools/dist_train.sh configs/oln_box_ships/round0/allship_cz_hybrid_lc25_r0.py 4
bash tools/dist_test_bbox_evalclass.sh configs/oln_box_ships/round0/allship_cz_hybrid_lc25_r0.py out/oln_box_ships/round0/allship_cz_hybrid_lc25_r0/latest.pth 'allship' 4
echo "DONE LC25"

bash tools/dist_train.sh configs/oln_box_ships/round0/allship_cz_hybrid_lc50_r0.py 4
bash tools/dist_test_bbox_evalclass.sh configs/oln_box_ships/round0/allship_cz_hybrid_lc50_r0.py out/oln_box_ships/round0/allship_cz_hybrid_lc50_r0/latest.pth 'allship' 4
echo "DONE LC50"

bash tools/dist_train.sh configs/oln_box_ships/round0/allship_cz_hybrid_lc75_r0.py 4
bash tools/dist_test_bbox_evalclass.sh configs/oln_box_ships/round0/allship_cz_hybrid_lc75_r0.py out/oln_box_ships/round0/allship_cz_hybrid_lc75_r0/latest.pth 'allship' 4
echo "DONE LC75"

bash tools/dist_train.sh configs/oln_box_ships/round0/allship_cz_hybrid_lc90_r0.py 4
bash tools/dist_test_bbox_evalclass.sh configs/oln_box_ships/round0/allship_cz_hybrid_lc90_r0.py out/oln_box_ships/round0/allship_cz_hybrid_lc90_r0/latest.pth 'allship' 4
echo "DONE LC90"

bash tools/dist_train.sh configs/oln_box_ships/round0/allship_cz_hybrid_lc1_r0.py 4
bash tools/dist_test_bbox_evalclass.sh configs/oln_box_ships/round0/allship_cz_hybrid_lc1_r0.py out/oln_box_ships/round0/allship_cz_hybrid_lc1_r0/latest.pth 'allship' 4
echo "DONE LC1"
