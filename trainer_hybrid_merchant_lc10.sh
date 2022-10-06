#!/bin/bash


### Round 0
bash tools/dist_train.sh configs/oln_box_ships/round0/merchant_cz_hybrid_lc10_r0.py 4
bash tools/dist_test_bbox_evalclass.sh configs/oln_box_ships/round0/merchant_cz_hybrid_lc10_r0.py out/oln_box_ships/round0/merchant_cz_hybrid_lc10_r0/latest.pth 'nonmerchant' 4
bash tools/dist_test_bbox_evalclass.sh configs/oln_box_ships/round0/merchant_cz_hybrid_lc10_r0.py out/oln_box_ships/round0/merchant_cz_hybrid_lc10_r0/latest.pth 'merchant' 4
bash tools/dist_test_bbox_evalclass.sh configs/oln_box_ships/round0/merchant_cz_hybrid_lc10_r0.py out/oln_box_ships/round0/merchant_cz_hybrid_lc10_r0/latest.pth 'allship' 4
echo "DONE ROUND0"


### Round 1
bash tools/dist_collect_preds.sh configs/oln_box_ships/round0/merchant_cz_hybrid_lc10_r0.py out/oln_box_ships/round0/merchant_cz_hybrid_lc10_r0/latest.pth 0.5 0 4
python -u tools/merge_annotations_by_percent.py data/ShipRSImageNet/COCO_Format/ShipRSImageNet_bbox_train_level_3.json out/oln_box_ships/round0/merchant_cz_hybrid_lc10_r0/preds_round0.bbox.json 'merchant' .30 --restricted

bash tools/dist_train.sh configs/oln_box_ships/round1/restricted_merchant_cz_hybrid_lc10_lateqflwbbl2_r1_p30.py 4
bash tools/dist_test_bbox_evalclass.sh configs/oln_box_ships/round1/restricted_merchant_cz_hybrid_lc10_lateqflwbbl2_r1_p30.py out/oln_box_ships/round1/restricted_merchant_cz_hybrid_lc10_lateqflwbbl2_r1_p30/latest.pth 'nonmerchant' 4
bash tools/dist_test_bbox_evalclass.sh configs/oln_box_ships/round1/restricted_merchant_cz_hybrid_lc10_lateqflwbbl2_r1_p30.py out/oln_box_ships/round1/restricted_merchant_cz_hybrid_lc10_lateqflwbbl2_r1_p30/latest.pth 'merchant_nou' 4
bash tools/dist_test_bbox_evalclass.sh configs/oln_box_ships/round1/restricted_merchant_cz_hybrid_lc10_lateqflwbbl2_r1_p30.py out/oln_box_ships/round1/restricted_merchant_cz_hybrid_lc10_lateqflwbbl2_r1_p30/latest.pth 'allship_nou' 4
echo "DONE ROUND1"


### Round 2
#bash tools/dist_collect_preds.sh configs/oln_box_ships/round1/restricted_merchant_cz_hybrid_lc10_lateqflwbbl2_r1_p30.py out/oln_box_ships/round1/restricted_merchant_cz_hybrid_lc10_lateqflwbbl2_r1_p30/latest.pth 0.5 1 4
#python tools/merge_annotations_by_percent.py data/ShipRSImageNet/COCO_Format/ShipRSImageNet_bbox_train_level_3.json out/oln_box_ships/round1/restricted_merchant_cz_hybrid_lc10_lateqflwbbl2_r1_p30/preds_round1.bbox.json 'merchant' .30 --restricted

#bash tools/dist_train.sh configs/oln_box_ships/round2/restricted_merchant_cz_hybrid_lc10_lateqflwbbl2_r2_p30_p30.py 4
#bash tools/dist_test_bbox_evalclass.sh configs/oln_box_ships/round2/restricted_merchant_cz_hybrid_lc10_lateqflwbbl2_r2_p30_p30.py out/oln_box_ships/round2/restricted_merchant_cz_hybrid_lc10_lateqflwbbl2_r2_p30_p30/latest.pth 'nonmerchant' 4
#bash tools/dist_test_bbox_evalclass.sh configs/oln_box_ships/round2/restricted_merchant_cz_hybrid_lc10_lateqflwbbl2_r2_p30_p30.py out/oln_box_ships/round2/restricted_merchant_cz_hybrid_lc10_lateqflwbbl2_r2_p30_p30/latest.pth 'merchant_nou' 4
#bash tools/dist_test_bbox_evalclass.sh configs/oln_box_ships/round2/restricted_merchant_cz_hybrid_lc10_lateqflwbbl2_r2_p30_p30.py out/oln_box_ships/round2/restricted_merchant_cz_hybrid_lc10_lateqflwbbl2_r2_p30_p30/latest.pth 'allship_nou' 4
#echo "DONE ROUND2"


### Round 3
#bash tools/dist_collect_preds.sh configs/oln_box_ships/round2/restricted_merchant_cz_hybrid_lc10_lateqflwbbl2_r2_p30_p30.py out/oln_box_ships/round2/restricted_merchant_cz_hybrid_lc10_lateqflwbbl2_r2_p30_p30/latest.pth 0.5 2 4
#python tools/merge_annotations_by_percent.py data/ShipRSImageNet/COCO_Format/ShipRSImageNet_bbox_train_level_3.json out/oln_box_ships/round2/restricted_merchant_cz_hybrid_lc10_lateqflwbbl2_r2_p30_p30/preds_round2.bbox.json 'merchant' .30 --restricted

#bash tools/dist_train.sh configs/oln_box_ships/round3/restricted_merchant_cz_hybrid_lc10_lateqflwbbl2_r3_p30_p30_p30.py 4
#bash tools/dist_test_bbox_evalclass.sh configs/oln_box_ships/round3/restricted_merchant_cz_hybrid_lc10_lateqflwbbl2_r3_p30_p30_p30.py out/oln_box_ships/round3/restricted_merchant_cz_hybrid_lc10_lateqflwbbl2_r3_p30_p30_p30/latest.pth 'nonmerchant' 4
#bash tools/dist_test_bbox_evalclass.sh configs/oln_box_ships/round3/restricted_merchant_cz_hybrid_lc10_lateqflwbbl2_r3_p30_p30_p30.py out/oln_box_ships/round3/restricted_merchant_cz_hybrid_lc10_lateqflwbbl2_r3_p30_p30_p30/latest.pth 'merchant_nou' 4
#bash tools/dist_test_bbox_evalclass.sh configs/oln_box_ships/round3/restricted_merchant_cz_hybrid_lc10_lateqflwbbl2_r3_p30_p30_p30.py out/oln_box_ships/round3/restricted_merchant_cz_hybrid_lc10_lateqflwbbl2_r3_p30_p30_p30/latest.pth 'allship_nou' 4
#echo "DONE ROUND3"
