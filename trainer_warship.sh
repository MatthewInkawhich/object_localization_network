#!/bin/bash


### Round 1
python -u tools/merge_annotations_by_percent.py data/ShipRSImageNet/COCO_Format/ShipRSImageNet_bbox_train_level_3.json out/oln_box_ships/round0/warship_cz_r0/preds_round0.bbox.json 'warship' .30

bash tools/dist_train.sh configs/oln_box_ships/round1/warship_cz_lateqflwbbl2_r1_p30.py 4
bash tools/dist_test_bbox_evalclass.sh configs/oln_box_ships/round1/warship_cz_lateqflwbbl2_r1_p30.py out/oln_box_ships/round1/warship_cz_lateqflwbbl2_r1_p30/latest.pth 'nonwarship' 4
bash tools/dist_test_bbox_evalclass.sh configs/oln_box_ships/round1/warship_cz_lateqflwbbl2_r1_p30.py out/oln_box_ships/round1/warship_cz_lateqflwbbl2_r1_p30/latest.pth 'warship_nou' 4
bash tools/dist_test_bbox_evalclass.sh configs/oln_box_ships/round1/warship_cz_lateqflwbbl2_r1_p30.py out/oln_box_ships/round1/warship_cz_lateqflwbbl2_r1_p30/latest.pth 'all_nou' 4
echo "DONE ROUND1"


### Round 2
bash tools/dist_collect_preds.sh configs/oln_box_ships/round1/warship_cz_lateqflwbbl2_r1_p30.py out/oln_box_ships/round1/warship_cz_lateqflwbbl2_r1_p30/latest.pth 0.6 1 4
python tools/merge_annotations_by_percent.py data/ShipRSImageNet/COCO_Format/ShipRSImageNet_bbox_train_level_3.json out/oln_box_ships/round1/warship_cz_lateqflwbbl2_r1_p30/preds_round1.bbox.json 'warship' .30

bash tools/dist_train.sh configs/oln_box_ships/round2/warship_cz_lateqflwbbl2_r2_p30_p30.py 4
bash tools/dist_test_bbox_evalclass.sh configs/oln_box_ships/round2/warship_cz_lateqflwbbl2_r2_p30_p30.py out/oln_box_ships/round2/warship_cz_lateqflwbbl2_r2_p30_p30/latest.pth 'nonwarship' 4
bash tools/dist_test_bbox_evalclass.sh configs/oln_box_ships/round2/warship_cz_lateqflwbbl2_r2_p30_p30.py out/oln_box_ships/round2/warship_cz_lateqflwbbl2_r2_p30_p30/latest.pth 'warship_nou' 4
bash tools/dist_test_bbox_evalclass.sh configs/oln_box_ships/round2/warship_cz_lateqflwbbl2_r2_p30_p30.py out/oln_box_ships/round2/warship_cz_lateqflwbbl2_r2_p30_p30/latest.pth 'all_nou' 4
echo "DONE ROUND2"


### Round 3
bash tools/dist_collect_preds.sh configs/oln_box_ships/round2/warship_cz_lateqflwbbl2_r2_p30_p30.py out/oln_box_ships/round2/warship_cz_lateqflwbbl2_r2_p30_p30/latest.pth 0.6 2 4
python tools/merge_annotations_by_percent.py data/ShipRSImageNet/COCO_Format/ShipRSImageNet_bbox_train_level_3.json out/oln_box_ships/round2/warship_cz_lateqflwbbl2_r2_p30_p30/preds_round2.bbox.json 'warship' .30

bash tools/dist_train.sh configs/oln_box_ships/round3/warship_cz_lateqflwbbl2_r3_p30_p30_p30.py 4
bash tools/dist_test_bbox_evalclass.sh configs/oln_box_ships/round3/warship_cz_lateqflwbbl2_r3_p30_p30_p30.py out/oln_box_ships/round3/warship_cz_lateqflwbbl2_r3_p30_p30_p30/latest.pth 'nonwarship' 4
bash tools/dist_test_bbox_evalclass.sh configs/oln_box_ships/round3/warship_cz_lateqflwbbl2_r3_p30_p30_p30.py out/oln_box_ships/round3/warship_cz_lateqflwbbl2_r3_p30_p30_p30/latest.pth 'warship_nou' 4
bash tools/dist_test_bbox_evalclass.sh configs/oln_box_ships/round3/warship_cz_lateqflwbbl2_r3_p30_p30_p30.py out/oln_box_ships/round3/warship_cz_lateqflwbbl2_r3_p30_p30_p30/latest.pth 'all_nou' 4
echo "DONE ROUND3"
