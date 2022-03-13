#!/bin/bash

### OLN round0
#bash tools/dist_train.sh configs/oln_box/round0/voc_ssl_p05_s1_cz_2x_r0.py 4
#bash tools/dist_test_bbox.sh configs/oln_box/round0/voc_ssl_p05_s1_cz_2x_r0.py out/oln_box/round0/voc_ssl_p05_s1_cz_2x_r0/latest.pth 4
#echo "DONE: SSL P05"

#bash tools/dist_train.sh configs/oln_box/round0/voc_ssl_p10_s1_cz_2x_r0.py 4
#bash tools/dist_test_bbox.sh configs/oln_box/round0/voc_ssl_p10_s1_cz_2x_r0.py out/oln_box/round0/voc_ssl_p10_s1_cz_2x_r0/latest.pth 4
#echo "DONE: SSL P10"

bash tools/dist_train.sh configs/oln_box/round0/voc_ssl_p25_s1_cz_2x_r0.py 4
bash tools/dist_test_bbox.sh configs/oln_box/round0/voc_ssl_p25_s1_cz_2x_r0.py out/oln_box/round0/voc_ssl_p25_s1_cz_2x_r0/latest.pth 4
echo "DONE: SSL P25"

#bash tools/dist_train.sh configs/oln_box/round0/voc_ssl_p50_s1_cz_2x_r0.py 4
#bash tools/dist_test_bbox.sh configs/oln_box/round0/voc_ssl_p50_s1_cz_2x_r0.py out/oln_box/round0/voc_ssl_p50_s1_cz_2x_r0/latest.pth 4
#echo "DONE: SSL P50"
