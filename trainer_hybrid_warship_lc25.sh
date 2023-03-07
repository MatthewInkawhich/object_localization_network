#!/bin/bash


### Round 0
#bash tools/dist_train.sh configs/oln_box_ships/round0/warship_cz_hybrid_lc25_r0.py 4
#bash tools/dist_test_bbox_evalclass.sh configs/oln_box_ships/round0/warship_cz_hybrid_lc25_r0.py out/oln_box_ships/round0/warship_cz_hybrid_lc25_r0/latest.pth 'nonwarship' 4
#bash tools/dist_test_bbox_evalclass.sh configs/oln_box_ships/round0/warship_cz_hybrid_lc25_r0.py out/oln_box_ships/round0/warship_cz_hybrid_lc25_r0/latest.pth 'warship' 4
#bash tools/dist_test_bbox_evalclass.sh configs/oln_box_ships/round0/warship_cz_hybrid_lc25_r0.py out/oln_box_ships/round0/warship_cz_hybrid_lc25_r0/latest.pth 'allship' 4


### Round 1
#bash tools/dist_collect_preds.sh configs/oln_box_ships/round0/warship_cz_hybrid_lc25_r0.py out/oln_box_ships/round0/warship_cz_hybrid_lc25_r0/latest.pth 0.5 0 4
#python -u tools/merge_annotations_by_percent.py data/ShipRSImageNet/COCO_Format/ShipRSImageNet_bbox_train_level_3.json out/oln_box_ships/round0/warship_cz_hybrid_lc25_r0/preds_round0.bbox.json 'warship' .30 --restricted

bash tools/dist_train.sh configs/oln_box_ships/round1/restricted_warship_cz_hybrid_lc25_lateqflwbbl2_r1_p30.py 4
bash tools/dist_test_bbox_evalclass.sh configs/oln_box_ships/round1/restricted_warship_cz_hybrid_lc25_lateqflwbbl2_r1_p30.py out/oln_box_ships/round1/restricted_warship_cz_hybrid_lc25_lateqflwbbl2_r1_p30/latest.pth 'nonwarship' 4
bash tools/dist_test_bbox_evalclass.sh configs/oln_box_ships/round1/restricted_warship_cz_hybrid_lc25_lateqflwbbl2_r1_p30.py out/oln_box_ships/round1/restricted_warship_cz_hybrid_lc25_lateqflwbbl2_r1_p30/latest.pth 'warship_nou' 4
bash tools/dist_test_bbox_evalclass.sh configs/oln_box_ships/round1/restricted_warship_cz_hybrid_lc25_lateqflwbbl2_r1_p30.py out/oln_box_ships/round1/restricted_warship_cz_hybrid_lc25_lateqflwbbl2_r1_p30/latest.pth 'allship_nou' 4
echo "DONE ROUND1"


