#!/bin/bash


### Round 1
python -u tools/merge_annotations_by_percent.py data/ShipRSImageNet/COCO_Format/ShipRSImageNet_bbox_train_level_3.json out/oln_box_ships/round0/merchant_cz_r0/preds_round0.bbox.json 'merchant' .60

bash tools/dist_train.sh configs/oln_box_ships/round1/merchant_cz_lateqflwbbl2_r1_p60.py 4
bash tools/dist_test_bbox_evalclass.sh configs/oln_box_ships/round1/merchant_cz_lateqflwbbl2_r1_p60.py out/oln_box_ships/round1/merchant_cz_lateqflwbbl2_r1_p60/latest.pth 'nonmerchant' 4
bash tools/dist_test_bbox_evalclass.sh configs/oln_box_ships/round1/merchant_cz_lateqflwbbl2_r1_p60.py out/oln_box_ships/round1/merchant_cz_lateqflwbbl2_r1_p60/latest.pth 'merchant_nou' 4
bash tools/dist_test_bbox_evalclass.sh configs/oln_box_ships/round1/merchant_cz_lateqflwbbl2_r1_p60.py out/oln_box_ships/round1/merchant_cz_lateqflwbbl2_r1_p60/latest.pth 'allship_nou' 4
echo "DONE ROUND1"


### Round 2
bash tools/dist_collect_preds.sh configs/oln_box_ships/round1/merchant_cz_lateqflwbbl2_r1_p60.py out/oln_box_ships/round1/merchant_cz_lateqflwbbl2_r1_p60/latest.pth 0.6 1 4
python tools/merge_annotations_by_percent.py data/ShipRSImageNet/COCO_Format/ShipRSImageNet_bbox_train_level_3.json out/oln_box_ships/round1/merchant_cz_lateqflwbbl2_r1_p60/preds_round1.bbox.json 'merchant' .60

bash tools/dist_train.sh configs/oln_box_ships/round2/merchant_cz_lateqflwbbl2_r2_p60_p60.py 4
bash tools/dist_test_bbox_evalclass.sh configs/oln_box_ships/round2/merchant_cz_lateqflwbbl2_r2_p60_p60.py out/oln_box_ships/round2/merchant_cz_lateqflwbbl2_r2_p60_p60/latest.pth 'nonmerchant' 4
bash tools/dist_test_bbox_evalclass.sh configs/oln_box_ships/round2/merchant_cz_lateqflwbbl2_r2_p60_p60.py out/oln_box_ships/round2/merchant_cz_lateqflwbbl2_r2_p60_p60/latest.pth 'merchant_nou' 4
bash tools/dist_test_bbox_evalclass.sh configs/oln_box_ships/round2/merchant_cz_lateqflwbbl2_r2_p60_p60.py out/oln_box_ships/round2/merchant_cz_lateqflwbbl2_r2_p60_p60/latest.pth 'allship_nou' 4
echo "DONE ROUND2"


### Round 3
bash tools/dist_collect_preds.sh configs/oln_box_ships/round2/merchant_cz_lateqflwbbl2_r2_p60_p60.py out/oln_box_ships/round2/merchant_cz_lateqflwbbl2_r2_p60_p60/latest.pth 0.6 2 4
python tools/merge_annotations_by_percent.py data/ShipRSImageNet/COCO_Format/ShipRSImageNet_bbox_train_level_3.json out/oln_box_ships/round2/merchant_cz_lateqflwbbl2_r2_p60_p60/preds_round2.bbox.json 'merchant' .60

bash tools/dist_train.sh configs/oln_box_ships/round3/merchant_cz_lateqflwbbl2_r3_p60_p60_p60.py 4
bash tools/dist_test_bbox_evalclass.sh configs/oln_box_ships/round3/merchant_cz_lateqflwbbl2_r3_p60_p60_p60.py out/oln_box_ships/round3/merchant_cz_lateqflwbbl2_r3_p60_p60_p60/latest.pth 'nonmerchant' 4
bash tools/dist_test_bbox_evalclass.sh configs/oln_box_ships/round3/merchant_cz_lateqflwbbl2_r3_p60_p60_p60.py out/oln_box_ships/round3/merchant_cz_lateqflwbbl2_r3_p60_p60_p60/latest.pth 'merchant_nou' 4
bash tools/dist_test_bbox_evalclass.sh configs/oln_box_ships/round3/merchant_cz_lateqflwbbl2_r3_p60_p60_p60.py out/oln_box_ships/round3/merchant_cz_lateqflwbbl2_r3_p60_p60_p60/latest.pth 'allship_nou' 4
echo "DONE ROUND3"
