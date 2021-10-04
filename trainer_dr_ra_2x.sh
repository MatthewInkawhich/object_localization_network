#!/bin/bash

bash tools/dist_train.sh configs/oln_box/aug_discreterotate_2x.py 4
bash tools/dist_test_bbox.sh configs/oln_box/aug_discreterotate_2x.py out/oln_box/aug_discreterotate_2x/latest.pth 4

#bash tools/dist_train.sh configs/oln_box/aug_randomaffine_2x.py 4
#bash tools/dist_test_bbox.sh configs/oln_box/aug_randomaffine_2x.py out/oln_box/aug_randomaffine_2x/latest.pth 4
