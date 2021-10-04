#!/bin/bash

bash tools/dist_train.sh configs/oln_box/aug_discreterotate.py 4
bash tools/dist_test_bbox.sh configs/oln_box/aug_discreterotate.py out/oln_box/aug_discreterotate/latest.pth 4
