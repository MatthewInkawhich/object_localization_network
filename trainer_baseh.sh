#!/bin/bash

bash tools/dist_train.sh configs/oln_box/baselines/hcoco_oln_2x.py 4
bash tools/dist_test_bbox_evalclass.sh configs/oln_box/baselines/hcoco_oln_2x.py out/oln_box/baselines/hcoco_oln_2x/latest.pth 'nonhcoco' 4
bash tools/dist_test_bbox_evalclass.sh configs/oln_box/baselines/hcoco_oln_2x.py out/oln_box/baselines/hcoco_oln_2x/latest.pth 'hcoco' 4
bash tools/dist_test_bbox_evalclass.sh configs/oln_box/baselines/hcoco_oln_2x.py out/oln_box/baselines/hcoco_oln_2x/latest.pth 'all' 4

