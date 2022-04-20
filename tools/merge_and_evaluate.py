import argparse
import os
import warnings
import numpy as np
import matplotlib.pyplot as plt

import mmcv
import torch
from mmcv import Config, DictAction
from mmdet.datasets import (build_dataloader, build_dataset,
                            replace_ImageToTensor)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Merge two sets of predictions and evaluate')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('primary_preds', help='path to our primary predictions (from our STPN localization-based model)')
    parser.add_argument('secondary_preds', help='path to our secondary predictions (from our GT-focused model)')
    parser.add_argument('N', type=float, help='Percent of secondary preds to consider [0,1]')
    parser.add_argument('--secondary-score-metric', type=str, default='conf', help='score metric of secondary_preds ("iou" or "conf")')
    args = parser.parse_args()
    return args


def normalize_data(data, min_v, max_v):
    #print("data:", data, data.shape)
    #print("min_v:", min_v)
    #print("max_v:", max_v)
    #print("np.max(data):", np.max(data))
    #print("np.min(data):", np.min(data))
    normalized_data = (((data - np.min(data)) / (np.max(data) - np.min(data))) * (max_v - min_v)) + min_v
    # Sometimes max == min for FRCNN (voc5 & animal)
    np.nan_to_num(normalized_data, copy=False)  # Convert nans to 0s
    return normalized_data

def bb_intersection_over_union(boxA, boxB):
    """
    https://www.pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/
    """
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
    # return the intersection over union value
    return iou


def main():
    ### Parse args
    args = parse_args()
    assert args.N >= 0 and args.N <= 1.0, "Error: args.N should be in [0, 1]"
    assert args.secondary_score_metric == 'iou' or args.secondary_score_metric == 'conf', "Error: args.secondary-score-metric " \
        "must be either 'iou' or 'conf'"

    ### Load configs
    cfg = Config.fromfile(args.config)
    
    ### Load predictions
    print("Loading primary preds...")
    primary_preds_contents = mmcv.load(args.primary_preds)
    print("primary preds loaded!", len(primary_preds_contents))
    print("Loading secondary preds...")
    secondary_preds_contents = mmcv.load(args.secondary_preds)
    print("secondary preds loaded!", len(primary_preds_contents))

    ### Hyperparameters
    # Min overlap threshold for prediction agreement
    OVERLAP_THRESH = 0.72

    ### Merge predictions per image
    all_merged_predictions = []
    #for image_idx in range(len(primary_preds_contents)):
    for image_idx in [4]:
        #single_idx = 4
        current_pps = primary_preds_contents[image_idx][0]
        current_sps = secondary_preds_contents[image_idx][0]
        
        # Sample the top N% of Faster R-CNN predictions
        n_elements = round(current_sps.shape[0] * args.N)
        #print("n_elements:", n_elements)
        useful_current_sps = current_sps[:n_elements, :]

        # Normalize these top N preds into the top N range of STPN preds
        normalized_useful_current_sps = np.copy(useful_current_sps)
        normalized_useful_current_sps[:, -1] = normalize_data(useful_current_sps[:, -1], current_pps[n_elements][-1], current_pps[0][-1])
        
        # Find agreeing pairs
        #print("\nFinding agreeing pairs...")
        clean_current_pps = np.copy(current_pps)
        #matched_pps_idxs = []
        matched_sps_idxs = []
        agreeing_preds = []
        for i in range(normalized_useful_current_sps.shape[0]):
            sps_box = normalized_useful_current_sps[i][:4]
            sps_score = normalized_useful_current_sps[i][-1]
            for j in range(clean_current_pps.shape[0]):
                pps_box = clean_current_pps[j][:4]
                pps_score = clean_current_pps[j][-1]
                iou = bb_intersection_over_union(sps_box, pps_box)
                if iou >= OVERLAP_THRESH:
                    #matched_pps_idxs.append(j)
                    clean_current_pps = np.delete(clean_current_pps, [j], axis=0)
                    matched_sps_idxs.append(i)
                    agreeing_pred = pps_box.tolist()
                    agreeing_pred.append(max(pps_score, sps_score))
                    agreeing_preds.append(agreeing_pred)
                    break

#            # Find the maximally overlapping pp for this sp
#            max_iou = 0
#            max_iou_j = 0
#            for j in range(current_pps.shape[0]):
#                # Check if the jth pp is not already matched to a sp
#                if j not in matched_pps_idxs:
#                    pps_box = current_pps[j][:4]
#                    pps_score = current_pps[j][-1]
#                    # Compute iou between sps_box and pps_box
#                    iou = bb_intersection_over_union(sps_box, pps_box)
#                    if iou > max_iou:
#                        max_iou = iou
#                        max_iou_j = j
#                        max_iou_pps_box = pps_box
#                        max_iou_pps_score = pps_score
#            # If the max_iou is above a threshold, add the offending pred to agreeing_preds 
#            if max_iou >= OVERLAP_THRESH:
#                matched_pps_idxs.append(max_iou_j)
#                matched_sps_idxs.append(i)
#                agreeing_pred = max_iou_pps_box.tolist()
#                agreeing_pred.append(max_iou_pps_score + sps_score)
#                agreeing_preds.append(agreeing_pred)


        agreeing_preds = np.array(agreeing_preds)
        #print("\nagreeing_preds:", agreeing_preds, agreeing_preds.shape)
        #print("matched_pps_idxs:", len(matched_pps_idxs), len(list(set(matched_pps_idxs))))
        #print("matched_sps_idxs:", len(matched_sps_idxs), len(list(set(matched_sps_idxs))))

        # Remove potential duplicates
        # Sometimes different secondary predictions agree with the same primary pred
        #matched_pps_idxs = list(set(matched_pps_idxs))


        ### Merge primary, secondary, and agreeing preds
        #clean_current_pps = np.delete(current_pps, matched_pps_idxs, axis=0)
        clean_normalized_useful_current_sps = np.delete(normalized_useful_current_sps, matched_sps_idxs, axis=0)
        #print("\ncurrent_pps:", current_pps.shape)
        #print("clean_current_pps:", clean_current_pps.shape)
        #print("normalized_useful_current_sps:", normalized_useful_current_sps.shape)
        #print("clean_normalized_useful_current_sps:", clean_normalized_useful_current_sps.shape)

        # If we have no agreeing preds, don't concat them
        if agreeing_preds.shape == (0,):
            merged_preds = np.concatenate((clean_current_pps, clean_normalized_useful_current_sps), axis=0)
        else:
            merged_preds = np.concatenate((agreeing_preds, clean_current_pps, clean_normalized_useful_current_sps), axis=0)

        # Re-normalize scores between [0,1]
        # Don't think this is necessary, just to maintain consistency with original
        #normalized_merged_preds= np.copy(merged_preds)
        #normalized_merged_preds[:, -1] = normalize_data(merged_preds[:, -1], 0, 1)
        # Sort preds by score
        #sorted_normalized_merged_preds = normalized_merged_preds[normalized_merged_preds[:, -1].argsort()][::-1]
        sorted_merged_preds = merged_preds[merged_preds[:, -1].argsort()][::-1]
        #print("\nsorted_normalized_merged_preds:", sorted_normalized_merged_preds, sorted_normalized_merged_preds.shape)
        #exit()




        marker_size = 10
        ### Plot all
        #plt.scatter(list(range(current_pps.shape[0])), current_pps[:,-1], s=marker_size, label='STPN Predictions')
        #plt.scatter(list(range(current_sps.shape[0])), current_sps[:,-1], s=marker_size, label='Faster R-CNN Predictions')
        #plt.legend()
        #plt.ylim(-0.02, 1.02)
        #plt.xlabel('Prediction Rank')
        #plt.ylabel('Score')
        #plt.show()


        ### Plot useful / normalized useful
        #plt.scatter(list(range(current_pps.shape[0])), current_pps[:,-1], s=marker_size, label='STPN Predictions')
        #plt.scatter(list(range(current_sps.shape[0])), current_sps[:,-1], s=marker_size, label='Faster R-CNN Predictions')
        #plt.scatter(list(range(len(useful_current_sps))), useful_current_sps[:,-1], s=marker_size, label='Useful Faster R-CNN Predictions (Top {}%)'.format(round(args.N*100)))
        #plt.scatter(list(range(normalized_useful_current_sps.shape[0])), normalized_useful_current_sps[:,-1], s=marker_size, label='Normalized Useful Faster R-CNN Predictions (Top {}%)'.format(round(args.N*100)))
        #plt.legend()
        #plt.ylim(-0.02, 1.02)
        #plt.xlabel('Prediction Rank')
        #plt.ylabel('Score')
        #plt.show()
    
    
        ### Plot 
        sorted_agreeing_preds = agreeing_preds[agreeing_preds[:, -1].argsort()][::-1]
        plt.scatter(list(range(clean_current_pps.shape[0])), clean_current_pps[:,-1], s=marker_size, label='STPN Predictions')
        plt.scatter(list(range(clean_normalized_useful_current_sps.shape[0])), clean_normalized_useful_current_sps[:,-1], s=marker_size, color='r', label='Normalized Useful Faster R-CNN Predictions (Top {}%)'.format(round(args.N*100)))
        plt.scatter(list(range(sorted_agreeing_preds.shape[0])), sorted_agreeing_preds[:,-1], s=marker_size, color='k', label='Overlapping Predictions')
        plt.legend()
        plt.ylim(-0.02, 1.02)
        plt.xlabel('Prediction Rank')
        plt.ylabel('Score')
        plt.show()


        exit()




        #all_merged_predictions.append([sorted_normalized_merged_preds])
        all_merged_predictions.append([sorted_merged_preds])
        if image_idx % 10 == 0:
            print("merging... {} / {}".format(image_idx, len(primary_preds_contents)))


    ### Prepare test config
    # in case the test dataset is concatenated
    samples_per_gpu = 1
    if isinstance(cfg.data.test, dict):
        cfg.data.test.test_mode = True
        samples_per_gpu = cfg.data.test.pop('samples_per_gpu', 1)
        if samples_per_gpu > 1:
            # Replace 'ImageToTensor' to 'DefaultFormatBundle'
            cfg.data.test.pipeline = replace_ImageToTensor(
                cfg.data.test.pipeline)
    elif isinstance(cfg.data.test, list):
        for ds_cfg in cfg.data.test:
            ds_cfg.test_mode = True
        samples_per_gpu = max(
            [ds_cfg.pop('samples_per_gpu', 1) for ds_cfg in cfg.data.test])
        if samples_per_gpu > 1:
            for ds_cfg in cfg.data.test:
                ds_cfg.pipeline = replace_ImageToTensor(ds_cfg.pipeline)


    ### Eval on ID, OOD, ALL
    trainclass = cfg.data.test.train_class
    if 'baselines/' in args.config or 'round0/' in args.config:
        evalclasses = ['non'+trainclass, trainclass, 'all']
    else:
        evalclasses = ['non'+trainclass, trainclass+'_nou', 'all_nou']

    # Loop over evalclasses
    for evalclass in evalclasses:
        cfg.data.test.eval_class = evalclass
        print("\n\n cfg.data.test.eval_class:", cfg.data.test.eval_class)
        # Build dataset
        dataset = build_dataset(cfg.data.test)
        print("dataset:", len(dataset))

        # Evaluate merged predictions
        kwargs = {}
        eval_kwargs = cfg.get('evaluation', {}).copy()
        for key in ['interval', 'tmpdir', 'start', 'gpu_collect', 'save_best', 'rule']:
            eval_kwargs.pop(key, None)
        eval_kwargs.update(dict(metric='bbox', **kwargs))
        #print(dataset.evaluate(primary_preds_contents, **eval_kwargs))
        print(dataset.evaluate(all_merged_predictions, **eval_kwargs))


if __name__ == '__main__':
    main()
