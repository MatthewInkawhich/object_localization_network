# MatthewInkawhich

import argparse
import os
import warnings
import numpy as np

import mmcv
import torch
from mmcv import Config, DictAction
from mmcv.cnn import fuse_conv_bn
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint,
                         wrap_fp16_model)

from mmdet.apis import multi_gpu_test, single_gpu_test, multi_gpu_collect_preds, single_gpu_collect_preds
from mmdet.datasets import (build_dataloader, build_dataset,
                            replace_ImageToTensor)
from mmdet.models import build_detector


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


def parse_args():
    parser = argparse.ArgumentParser(
        description='Generate pseudolabels')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('score_thr', type=float, help='score threshold')
    parser.add_argument('round', type=int, help='self-training round')
    parser.add_argument('scale_jitter', type=float, help='scale jitter')
    parser.add_argument(
        '--fuse-conv-bn',
        action='store_true',
        help='Whether to fuse conv and bn, this will slightly increase'
        'the inference speed')
    parser.add_argument(
        '--gpu-collect',
        action='store_true',
        help='whether to use gpu to collect results.')
    parser.add_argument(
        '--tmpdir',
        help='tmp directory used for collecting results from multiple '
        'workers, available when gpu-collect is not specified')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


def main():
    args = parse_args()
    assert args.score_thr >= 0 and args.score_thr <= 1, "score_thr needs to be in range [0,1]"
    assert args.scale_jitter >= 0 and args.scale_jitter <= 1, "scale_jitter needs to be in range [0,1]"

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    # import modules from string list.
    if cfg.get('custom_imports', None):
        from mmcv.utils import import_modules_from_strings
        import_modules_from_strings(**cfg['custom_imports'])

    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    cfg.model.pretrained = None
    if cfg.model.get('neck'):
        if isinstance(cfg.model.neck, list):
            for neck_cfg in cfg.model.neck:
                if neck_cfg.get('rfp_backbone'):
                    if neck_cfg.rfp_backbone.get('pretrained'):
                        neck_cfg.rfp_backbone.pretrained = None
        elif cfg.model.neck.get('rfp_backbone'):
            if cfg.model.neck.rfp_backbone.get('pretrained'):
                cfg.model.neck.rfp_backbone.pretrained = None

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

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    rank, _ = get_dist_info()
    if rank == 0:
        print("args.score_thr:", args.score_thr)
        print("args.round:", args.round)
        print("args.scale_jitter:", args.scale_jitter)

    # build the model and load checkpoint
    cfg.model.train_cfg = None
    model = build_detector(cfg.model, test_cfg=cfg.get('test_cfg'))
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
    if args.fuse_conv_bn:
        model = fuse_conv_bn(model)

    if rank == 0:
        print("\nmodel:\n", model)
    # old versions did not save class info in checkpoints, this walkaround is
    # for backward compatibility
    if 'CLASSES' in checkpoint['meta']:
        model.CLASSES = checkpoint['meta']['CLASSES']
    else:
        model.CLASSES = dataset.CLASSES




    ################################################################################
    ### COLLECT PREDS ON MULTIPLE DIFFERENT IMG_SCALES INTO ALL_OUTPUTS
    ################################################################################
    msfa_idx = -1
    for j in range(len(cfg.data.test.pipeline)):
        if cfg.data.test.pipeline[j].type == "MultiScaleFlipAug":
            msfa_idx = j
            break
    if msfa_idx < 0:
        print("Error: could not find MultiScaleFlipAug in test_pipeline")
        exit()
    resize_idx = -1
    for j in range(len(cfg.data.test.pipeline[msfa_idx].transforms)):
        if cfg.data.test.pipeline[msfa_idx].transforms[j].type == "Resize":
            resize_idx = j
            break
    if resize_idx < 0:
        print("Error: could not find Resize in test_pipeline.transforms")
        exit()

    # Define set of img_scales
    default_img_scale = cfg.data.test.pipeline[msfa_idx].img_scale
    if rank == 0:
        print("default_img_scale:", default_img_scale)
    default_width = default_img_scale[0]
    default_height = default_img_scale[1]
    img_scales = [
        (round(default_width), round(default_height)), # default
        (round(default_width*(1+args.scale_jitter)), round(default_height*(1-args.scale_jitter))), # wide
        (round(default_width*(1-args.scale_jitter)), round(default_height*(1+args.scale_jitter))), # narrow
        (round(default_width*(1-args.scale_jitter)), round(default_height*(1-args.scale_jitter))), # small
        (round(default_width*(1+args.scale_jitter)), round(default_height*(1+args.scale_jitter))), # large
    ]

    all_outputs = []
    # Iterative over img_scales
    for img_scale in img_scales:
        # Change img_scale in config
        cfg.data.test.pipeline[msfa_idx].img_scale = img_scale
        # Make sure keep_ratio is set to False
        cfg.data.test.pipeline[msfa_idx].transforms[resize_idx].keep_ratio = False

        # Build the dataset/dataloader
        # Remember, we want to run a "test" on the training data
        cfg.data.test.ann_file = cfg.data.test.ann_file.replace("val2017", "train2017")
        cfg.data.test.img_prefix = cfg.data.test.img_prefix.replace("val2017", "train2017")
        dataset = build_dataset(cfg.data.test)
        if rank == 0:
            print("\n\nCurrent img_scale:", img_scale)
            print("cfg.data.test:", cfg.data.test)
            print("dataset length:", len(dataset))
        data_loader = build_dataloader(
            dataset,
            samples_per_gpu=samples_per_gpu,
            workers_per_gpu=cfg.data.workers_per_gpu,
            dist=distributed,
            shuffle=False)

        # Run inference on all images
        outputs = []
        if not distributed:
            model = MMDataParallel(model, device_ids=[0])
            outputs = single_gpu_collect_preds(model, data_loader, score_thr=args.score_thr)
        else:
            model = MMDistributedDataParallel(
                model.cuda(),
                device_ids=[torch.cuda.current_device()],
                broadcast_buffers=False)
            outputs = multi_gpu_collect_preds(model, data_loader, args.score_thr, args.tmpdir, args.gpu_collect)

        #print("\n\noutputs:", img_scale)
        #for i in range(len(outputs)):
        #    print(i, outputs[i])
        #exit()
        if rank == 0:
            total_preds = 0
            for i in range(len(outputs)):
                total_preds += outputs[i][0].shape[0]
            print(f"\ntotal_preds (img_scale:{img_scale}):", total_preds)

        all_outputs.append(outputs)


    ################################################################################
    ### FILTER OUT NON-PERSISTENT PREDS
    ################################################################################
    # Run this on just one process
    if rank == 0:
        num_img_scales = len(img_scales)
        num_images = len(all_outputs[0])
        print("\n\nFiltering non-persistent predictions...")
        print("num_img_scales:", num_img_scales)
        print("num_images:", num_images)
        persistent_outputs = []
        total_original_preds = 0
        total_persistent_preds = 0

        ## TMP
        thr = 0.8
        pps_over_thr_before = 0


        for img_idx in range(num_images):
            # IDEA: Starting by assuming all the baseline preds (noaug) are persistent.
            # Then, for each current persistent_pred, iterate over the preds from the other augs
            # and compute IoU. If the max IoU found is < 0.7, remove the prediction from
            # persistent_preds. At the end of this process we are left with good persistent_preds.

            # Start by assuming the preds from the first aug (baseline) are persistent
            persistent_preds = np.copy(all_outputs[0][img_idx][0])
            total_original_preds += persistent_preds.shape[0]
            #print("\n\nInitial persistent_preds:", persistent_preds, img_idx) 

            ### TMP
            pps_over_thr_before += sum(persistent_preds[:, -1] >= thr)

            # Iterate over remaining aug_idxs
            for aug_idx in range(1, num_img_scales):
                preds_to_delete = []
                # Iterate over each current persistent pred
                for pp_idx in range(persistent_preds.shape[0]):
                    # Compute overlap with every pred in aug_idx preds
                    iou_overlaps = [bb_intersection_over_union(persistent_preds[pp_idx][:4].tolist(), all_outputs[aug_idx][img_idx][0][j][:4].tolist()) \
                        for j in range(all_outputs[aug_idx][img_idx][0].shape[0])]
                    #print("iou_overlaps:", iou_overlaps)
                    if len(iou_overlaps) > 0:
                        # Get max_idx, max of iou_overlaps
                        max_iou, max_iou_idx = max(iou_overlaps), iou_overlaps.index(max(iou_overlaps))
                        #print("max_iou:", max_iou)
                        #print("max_iou_idx:", max_iou_idx)
                        # Update persistent_preds
                        if max_iou < 0.7:
                            preds_to_delete.append(pp_idx)
                        #print("preds_to_delete:", preds_to_delete)
                    else:
                        preds_to_delete.append(pp_idx)

                # We can now delete the elements at preds_to_delete so we don't have to consider them next round
                persistent_preds = np.delete(persistent_preds, preds_to_delete, axis=0)
                #print("persistent_preds after aug_idx:{}".format(aug_idx), persistent_preds) 

            # We now have good persistent_preds for this image
            persistent_outputs.append([persistent_preds])
            total_persistent_preds += persistent_preds.shape[0]

        
        #print("\n\npersistent_outputs:")
        #for i in range(len(persistent_outputs)):
        #    print(i, persistent_outputs[i])
        #exit()
        print("total_original_preds:", total_original_preds)
        print("total_persistent_preds:", total_persistent_preds)


        ## TMP
        pps_over_thr_after = 0
        for i in range(len(persistent_outputs)):
            for j in range(persistent_outputs[i][0].shape[0]):
                if persistent_outputs[i][0][j][-1] >= thr:
                    pps_over_thr_after += 1
            
        print("thr:", thr)
        print("pps_over_thr (before):", pps_over_thr_before)
        print("pps_over_thr (after):", pps_over_thr_after)

        ################################################################################
        ### OUTPUT ROBUST PREDS TO JSON
        ################################################################################
        # Output results to json
        str_scale_jitter = "{:.2f}".format(args.scale_jitter).split('.')[-1]
        jsonfile_prefix = os.path.join(args.checkpoint.replace(args.checkpoint.split('/')[-1], ""), f"robustpreds{str_scale_jitter}_round{args.round}")
        print("\njsonfile_prefix:", jsonfile_prefix)
        dataset.format_results(persistent_outputs, jsonfile_prefix=jsonfile_prefix)
        print("\nFile written!")


if __name__ == '__main__':
    main()
