#!/bin/bash


### Round 1
bash tools/dist_collect_preds.sh configs/oln_box/round0/voc_ssl_p50_s1_cz_2x_r0.py out/oln_box/round0/voc_ssl_p50_s1_cz_2x_r0/latest.pth 0.6 0 4
python tools/merge_annotations_by_percent.py data/coco/ssl_annotations/ssl_p50_s1_annotations.json out/oln_box/round0/voc_ssl_p50_s1_cz_2x_r0/preds_round0.bbox.json 246407 .30

bash tools/dist_train.sh configs/oln_box/round1/voc_ssl_p50_s1_cz_lateqflwbbl2_noft_2x_r1_p30.py 4
bash tools/dist_test_bbox.sh configs/oln_box/round1/voc_ssl_p50_s1_cz_lateqflwbbl2_noft_2x_r1_p30.py out/oln_box/round1/voc_ssl_p50_s1_cz_lateqflwbbl2_noft_2x_r1_p30/latest.pth 4
echo "DONE ROUND1"


### Round 2
bash tools/dist_collect_preds.sh configs/oln_box/round1/voc_ssl_p50_s1_cz_lateqflwbbl2_noft_2x_r1_p30.py out/oln_box/round1/voc_ssl_p50_s1_cz_lateqflwbbl2_noft_2x_r1_p30/latest.pth 0.6 1 4
python tools/merge_annotations_by_percent.py data/coco/ssl_annotations/ssl_p50_s1_annotations.json out/oln_box/round1/voc_ssl_p50_s1_cz_lateqflwbbl2_noft_2x_r1_p30/preds_round1.bbox.json 246407 .30

bash tools/dist_train.sh configs/oln_box/round2/voc_ssl_p50_s1_cz_lateqflwbbl2_noft_2x_r2_p30_p30.py 4
bash tools/dist_test_bbox.sh configs/oln_box/round2/voc_ssl_p50_s1_cz_lateqflwbbl2_noft_2x_r2_p30_p30.py out/oln_box/round2/voc_ssl_p50_s1_cz_lateqflwbbl2_noft_2x_r2_p30_p30/latest.pth 4
echo "DONE ROUND2"

