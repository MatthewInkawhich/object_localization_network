#!/bin/bash


### Round 1
python -u tools/merge_annotations_by_percent.py data/ShipRSImageNet/COCO_Format/ShipRSImageNet_bbox_train_level_3.json out/oln_box_ships/round0/warship_cz_r0/preds_round0.bbox.json 'warship' 1.0 --oracle-anns data/ShipRSImageNet/COCO_Format/ShipRSImageNet_bbox_train_level_3.json --oracle-filter-percent 0.5

bash tools/dist_train.sh configs/oln_box_ships/round1/filter50_warship_cz_lateqflwbbl2_noft_r1_p100.py 4
bash tools/dist_test_bbox_evalclass.sh configs/oln_box_ships/round1/filter50_warship_cz_lateqflwbbl2_noft_r1_p100.py out/oln_box_ships/round1/filter50_warship_cz_lateqflwbbl2_noft_r1_p100/latest.pth 'nonwarship' 4
bash tools/dist_test_bbox_evalclass.sh configs/oln_box_ships/round1/filter50_warship_cz_lateqflwbbl2_noft_r1_p100.py out/oln_box_ships/round1/filter50_warship_cz_lateqflwbbl2_noft_r1_p100/latest.pth 'warship_nou' 4
bash tools/dist_test_bbox_evalclass.sh configs/oln_box_ships/round1/filter50_warship_cz_lateqflwbbl2_noft_r1_p100.py out/oln_box_ships/round1/filter50_warship_cz_lateqflwbbl2_noft_r1_p100/latest.pth 'all_nou' 4
echo "DONE ROUND1"


### Round 2
bash tools/dist_collect_preds.sh configs/oln_box_ships/round1/filter50_warship_cz_lateqflwbbl2_noft_r1_p100.py out/oln_box_ships/round1/filter50_warship_cz_lateqflwbbl2_noft_r1_p100/latest.pth 0.6 1 4
python tools/merge_annotations_by_percent.py data/ShipRSImageNet/COCO_Format/ShipRSImageNet_bbox_train_level_3.json out/oln_box_ships/round1/filter50_warship_cz_lateqflwbbl2_noft_r1_p100/preds_round1.bbox.json 'warship' 1.0 --oracle-anns data/ShipRSImageNet/COCO_Format/ShipRSImageNet_bbox_train_level_3.json --oracle-filter-percent 0.5

bash tools/dist_train.sh configs/oln_box_ships/round2/filter50_warship_cz_lateqflwbbl2_noft_r2_p100_p100.py 4
bash tools/dist_test_bbox_evalclass.sh configs/oln_box_ships/round2/filter50_warship_cz_lateqflwbbl2_noft_r2_p100_p100.py out/oln_box_ships/round2/filter50_warship_cz_lateqflwbbl2_noft_r2_p100_p100/latest.pth 'nonwarship' 4
bash tools/dist_test_bbox_evalclass.sh configs/oln_box_ships/round2/filter50_warship_cz_lateqflwbbl2_noft_r2_p100_p100.py out/oln_box_ships/round2/filter50_warship_cz_lateqflwbbl2_noft_r2_p100_p100/latest.pth 'warship_nou' 4
bash tools/dist_test_bbox_evalclass.sh configs/oln_box_ships/round2/filter50_warship_cz_lateqflwbbl2_noft_r2_p100_p100.py out/oln_box_ships/round2/filter50_warship_cz_lateqflwbbl2_noft_r2_p100_p100/latest.pth 'all_nou' 4
echo "DONE ROUND2"

