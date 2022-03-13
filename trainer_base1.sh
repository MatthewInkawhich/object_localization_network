#!/bin/bash

### OLN baselines
bash tools/dist_train.sh configs/oln_box/baselines/voc_ssl_p05_s1_oln.py 4
bash tools/dist_test_bbox.sh configs/oln_box/baselines/voc_ssl_p05_s1_oln.py out/oln_box/baselines/voc_ssl_p05_s1_oln/latest.pth 4
echo "DONE: SSL P05"

bash tools/dist_train.sh configs/oln_box/baselines/voc_ssl_p10_s1_oln.py 4
bash tools/dist_test_bbox.sh configs/oln_box/baselines/voc_ssl_p10_s1_oln.py out/oln_box/baselines/voc_ssl_p10_s1_oln/latest.pth 4
echo "DONE: SSL P10"

bash tools/dist_train.sh configs/oln_box/baselines/voc_ssl_p25_s1_oln.py 4
bash tools/dist_test_bbox.sh configs/oln_box/baselines/voc_ssl_p25_s1_oln.py out/oln_box/baselines/voc_ssl_p25_s1_oln/latest.pth 4
echo "DONE: SSL P25"

bash tools/dist_train.sh configs/oln_box/baselines/voc_ssl_p50_s1_oln.py 4
bash tools/dist_test_bbox.sh configs/oln_box/baselines/voc_ssl_p50_s1_oln.py out/oln_box/baselines/voc_ssl_p50_s1_oln/latest.pth 4
echo "DONE: SSL P50"
