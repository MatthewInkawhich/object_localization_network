import argparse
import os
from pathlib import Path
import random
import numpy as np

import torch
import mmcv
from mmcv import Config

from mmdet.core.utils import mask2ndarray
from mmdet.core.visualization import imshow_det_bboxes
from mmdet.datasets.builder import build_dataset


def parse_args():
    parser = argparse.ArgumentParser(description='Browse a dataset')
    parser.add_argument('config', help='train config file path')
    parser.add_argument(
        '--skip-type',
        type=str,
        nargs='+',
        default=['DefaultFormatBundle', 'Normalize', 'Collect'],
        help='skip some useless pipeline')
    parser.add_argument(
        '--output-dir',
        default=None,
        type=str,
        help='If there is no display interface, you can save it')
    parser.add_argument('--not-show', default=False, action='store_true')
    parser.add_argument(
        '--show-interval',
        type=float,
        default=2,
        help='the interval of show (s)')
    args = parser.parse_args()
    return args


def retrieve_data_cfg(config_path, skip_type):
    cfg = Config.fromfile(config_path)
    train_data_cfg = cfg.data.train
    mixup_data_cfg = cfg.data.mixup
    train_data_cfg['pipeline'] = [
        x for x in train_data_cfg.pipeline if x['type'] not in skip_type
    ]
    mixup_data_cfg['pipeline'] = [
        x for x in mixup_data_cfg.pipeline if x['type'] not in skip_type
    ]
    return cfg


def main():
    args = parse_args()

    cfg = retrieve_data_cfg(args.config, args.skip_type)
    # TRAIN
    print("\n\ncfg.data.train.items():")
    for k,v in cfg.data.train.items():
        print(k,v)
    dataset = build_dataset(cfg.data.train)
    print("len(dataset):", len(dataset))

    # MIX
    print("\n\ncfg.data.mixup.items():")
    for k,v in cfg.data.mixup.items():
        print(k,v)
    dataset_mixup = build_dataset(cfg.data.mixup)
    #subset_size = 30
    #dataset_mixup = torch.utils.data.random_split(dataset_mixup, [subset_size, len(dataset_mixup)-subset_size])[0]
    print("len(dataset_mixup):", len(dataset_mixup))


    class_names = dataset.CLASSES


    progress_bar = mmcv.ProgressBar(len(dataset))

    for idx, item in enumerate(dataset):
        filename = os.path.join(args.output_dir,
                                Path(item['filename']).name
                                ) if args.output_dir is not None else None
        gt_masks = item.get('gt_masks', None)
        if gt_masks is not None:
            gt_masks = mask2ndarray(gt_masks)
        
        print("\n\nITEM:")
        for k,v in item.items():
            print("\n",k,v)

        # Get mixup sample
        mixup_idx = random.randint(0, len(dataset_mixup)-1)
        item_mixup = dataset_mixup[mixup_idx]
        print("\n\nITEM MIX:")
        for k,v in item_mixup.items():
            print("\n",k,v)

        
        # Perform the mixup
        print("\n\n\n")
        print("\nitem[img]:", item['img'], item['img'].shape)
        print("\nitem_mixup[img]:", item_mixup['img'], item_mixup['img'].shape)

        mixup_ratio = 0.5
        new_img = mixup_ratio * item['img'] + (1 - mixup_ratio) * item_mixup['img']
        # Need to cast back to uint8
        new_img = new_img.astype(np.uint8)
        print("\nnew_img:", new_img, new_img.shape, new_img.dtype)

        print("\nitem[gt_bboxes]:", item['gt_bboxes'], item['gt_bboxes'].shape)
        print("\nitem_mixup[gt_bboxes]:", item_mixup['gt_bboxes'], item_mixup['gt_bboxes'].shape)
        new_gt_bboxes = np.concatenate((item['gt_bboxes'], item_mixup['gt_bboxes']))
        print("\nnew_gt_bboxes:", new_gt_bboxes, new_gt_bboxes.shape)

        print("\nitem[gt_labels]:", item['gt_labels'], item['gt_labels'].shape)
        print("\nitem_mixup[gt_labels]:", item_mixup['gt_labels'], item_mixup['gt_labels'].shape)
        new_gt_labels = np.concatenate((item['gt_labels'], item_mixup['gt_labels']))
        print("\nnew_gt_labels:", new_gt_labels, new_gt_labels.shape)


        

        imshow_det_bboxes(
            new_img,
            new_gt_bboxes,
            new_gt_labels,
            gt_masks,
            #class_names=dataset.CLASSES,
            class_names=class_names,
            show=not args.not_show,
            wait_time=args.show_interval,
            out_file=filename,
            bbox_color=(255, 102, 61),
            text_color=(255, 102, 61))

        progress_bar.update()


if __name__ == '__main__':
    main()
