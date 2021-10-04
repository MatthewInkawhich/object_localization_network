#!/bin/bash

bash tools/dist_train.sh configs/oln_box/voc_split_2x.py 4
bash tools/dist_test_bbox.sh configs/oln_box/voc_split_2x.py out/oln_box/voc_split_2x/latest.pth 4

bash tools/dist_train.sh configs/oln_box/aug_cropzoom_2x.py 4
bash tools/dist_test_bbox.sh configs/oln_box/aug_cropzoom_2x.py out/oln_box/aug_cropzoom_2x/latest.pth 4
