#!/bin/bash

### Faster R-CNN baselines
bash tools/dist_train.sh configs/oln_box/baselines/voc5_class_agn_faster_rcnn.py 4
bash tools/dist_test_bbox.sh configs/oln_box/baselines/voc5_class_agn_faster_rcnn.py out/oln_box/baselines/voc5_class_agn_faster_rcnn/latest.pth 4


