import argparse
import os
from pathlib import Path

import numpy as np
import torch
import mmcv
from mmcv import Config

from mmdet.core.utils import mask2ndarray
from mmdet.core.visualization import imshow_det_bboxes, imshow_gt_det_bboxes
from mmdet.datasets.builder import build_dataset


def parse_args():
    parser = argparse.ArgumentParser(description='Browse a dataset')
    parser.add_argument('config', help='train config file path')
    parser.add_argument(
        '--skip-type',
        type=str,
        nargs='+',
        default=['RandomFlip', 'MinIoURandomCrop', 'DefaultFormatBundle', 'Normalize', 'Collect'],
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
    train_data_cfg['pipeline'] = [
        x for x in train_data_cfg.pipeline if x['type'] not in skip_type
    ]
    return cfg




def main():
    args = parse_args()

    # TRAIN
    cfg = retrieve_data_cfg(args.config, args.skip_type)
    print("\ncfg.data.train.items():")
    for k,v in cfg.data.train.items():
        print(k,v)
    dataset = build_dataset(cfg.data.train)

    class_names = dataset.CLASSES

    #subset_size = 1000
    #dataset = torch.utils.data.random_split(dataset, [subset_size, len(dataset)-subset_size])[0]

    progress_bar = mmcv.ProgressBar(len(dataset))

    for i, item in enumerate(dataset):
        filename = os.path.join(args.output_dir,
                                Path(item['filename']).name
                                ) if args.output_dir is not None else None

        gt_masks = item.get('gt_masks', None)
        if gt_masks is not None:
            gt_masks = mask2ndarray(gt_masks)


        print("\n\n", i)

        # Only consider certain examples...
        gt_count = 0
        pl_count = 0
        pl_check = True
        for j in range(len(item['gt_scores'])):
            if item['gt_scores'][j] < 1:
                pl_count += 1
            else:
                gt_count += 1
        if gt_count < 1 or pl_count < 3:
            print("SKIPPING...")    
            continue

        #for k,v in item.items():
        #    print("\n",k,v)

        # Separate original GT from pseudo-labels
        gt_bboxes = []
        gt_labels = []
        pl_bboxes = []
        pl_labels = []
        for j in range(len(item['gt_scores'])):
            if item['gt_scores'][j] < 1:
                pl_bboxes.append(np.append(item['gt_bboxes'][j], item['gt_scores'][j]))
                pl_labels.append(item['gt_labels'][j])
            else:
                gt_bboxes.append(item['gt_bboxes'][j])
                gt_labels.append(item['gt_labels'][j])

        gt_bboxes = np.stack(gt_bboxes, axis=0)
        gt_labels = np.array(gt_labels)
        print("gt_bboxes:", gt_bboxes)
        print("gt_labels:", gt_labels)
        pl_bboxes = np.stack(pl_bboxes, axis=0)
        pl_labels = np.array(pl_labels)
        print("pl_bboxes:", pl_bboxes)
        print("pl_labels:", pl_labels)
        
        # Plot GT boxes
        thickness = 2
        fig_size = (8, 6)
        img = imshow_det_bboxes(
            item['img'],
            gt_bboxes,
            gt_labels,
            bbox_color=(255,17,0),
            text_color=(255,17,0),
            thickness=thickness,
            fig_size=fig_size,
            show=False,

        )

        # Plot PL boxes
        imshow_det_bboxes(
            img,
            pl_bboxes,
            pl_labels,
            bbox_color=(17, 255, 0),
            text_color=(17, 255, 0),
            thickness=thickness,
            fig_size=fig_size,
            #show=(not args.not_show)
            wait_time=args.show_interval,
            out_file=filename,
        )


#        imshow_det_bboxes(
#            item['img'],
#            item['gt_bboxes'],
#            item['gt_labels'],
#            gt_masks,
#            #class_names=class_names,
#            wait_time=args.show_interval,
#            out_file=filename,
#            bbox_color=(255, 102, 61),
#            text_color=(255, 102, 61))

        progress_bar.update()


if __name__ == '__main__':
    main()
