#!/bin/bash


### Round 1
python -u tools/merge_annotations_by_percent.py data/coco/annotations/instances_train2017.json out/oln_box/round0/animal_cz_2x_r0/preds_round0.bbox.json 'animal' .30 --restricted

bash tools/dist_train.sh configs/oln_box/round1/restricted_animal_cz_lateqflwbbl2_2x_r1_p30.py 4
bash tools/dist_test_bbox_evalclass.sh configs/oln_box/round1/restricted_animal_cz_lateqflwbbl2_2x_r1_p30.py out/oln_box/round1/restricted_animal_cz_lateqflwbbl2_2x_r1_p30/latest.pth 'nonanimal' 4
bash tools/dist_test_bbox_evalclass.sh configs/oln_box/round1/restricted_animal_cz_lateqflwbbl2_2x_r1_p30.py out/oln_box/round1/restricted_animal_cz_lateqflwbbl2_2x_r1_p30/latest.pth 'animal_nou' 4
bash tools/dist_test_bbox_evalclass.sh configs/oln_box/round1/restricted_animal_cz_lateqflwbbl2_2x_r1_p30.py out/oln_box/round1/restricted_animal_cz_lateqflwbbl2_2x_r1_p30/latest.pth 'all_nou' 4
echo "DONE ROUND1"


### Round 2
bash tools/dist_collect_preds.sh configs/oln_box/round1/restricted_animal_cz_lateqflwbbl2_2x_r1_p30.py out/oln_box/round1/restricted_animal_cz_lateqflwbbl2_2x_r1_p30/latest.pth 0.6 1 4
python tools/merge_annotations_by_percent.py data/coco/annotations/instances_train2017.json out/oln_box/round1/restricted_animal_cz_lateqflwbbl2_2x_r1_p30/preds_round1.bbox.json 'animal' .30 --restricted

bash tools/dist_train.sh configs/oln_box/round2/restricted_animal_cz_lateqflwbbl2_2x_r2_p30_p30.py 4
bash tools/dist_test_bbox_evalclass.sh configs/oln_box/round2/restricted_animal_cz_lateqflwbbl2_2x_r2_p30_p30.py out/oln_box/round2/restricted_animal_cz_lateqflwbbl2_2x_r2_p30_p30/latest.pth 'nonanimal' 4
bash tools/dist_test_bbox_evalclass.sh configs/oln_box/round2/restricted_animal_cz_lateqflwbbl2_2x_r2_p30_p30.py out/oln_box/round2/restricted_animal_cz_lateqflwbbl2_2x_r2_p30_p30/latest.pth 'animal_nou' 4
bash tools/dist_test_bbox_evalclass.sh configs/oln_box/round2/restricted_animal_cz_lateqflwbbl2_2x_r2_p30_p30.py out/oln_box/round2/restricted_animal_cz_lateqflwbbl2_2x_r2_p30_p30/latest.pth 'all_nou' 4
echo "DONE ROUND2"


### Round 3
bash tools/dist_collect_preds.sh configs/oln_box/round2/restricted_animal_cz_lateqflwbbl2_2x_r2_p30_p30.py out/oln_box/round2/restricted_animal_cz_lateqflwbbl2_2x_r2_p30_p30/latest.pth 0.6 2 4
python tools/merge_annotations_by_percent.py data/coco/annotations/instances_train2017.json out/oln_box/round2/restricted_animal_cz_lateqflwbbl2_2x_r2_p30_p30/preds_round2.bbox.json 'animal' .30 --restricted

bash tools/dist_train.sh configs/oln_box/round3/restricted_animal_cz_lateqflwbbl2_2x_r3_p30_p30_p30.py 4
bash tools/dist_test_bbox_evalclass.sh configs/oln_box/round3/restricted_animal_cz_lateqflwbbl2_2x_r3_p30_p30_p30.py out/oln_box/round3/restricted_animal_cz_lateqflwbbl2_2x_r3_p30_p30_p30/latest.pth 'nonanimal' 4
bash tools/dist_test_bbox_evalclass.sh configs/oln_box/round3/restricted_animal_cz_lateqflwbbl2_2x_r3_p30_p30_p30.py out/oln_box/round3/restricted_animal_cz_lateqflwbbl2_2x_r3_p30_p30_p30/latest.pth 'animal_nou' 4
bash tools/dist_test_bbox_evalclass.sh configs/oln_box/round3/restricted_animal_cz_lateqflwbbl2_2x_r3_p30_p30_p30.py out/oln_box/round3/restricted_animal_cz_lateqflwbbl2_2x_r3_p30_p30_p30/latest.pth 'all_nou' 4
echo "DONE ROUND3"
