import argparse
import os
from pathlib import Path

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
    parser.add_argument('--outlier', default=False, action='store_true')
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

def retrieve_data_cfg_outlier(config_path, skip_type):
    cfg = Config.fromfile(config_path)
    train_data_cfg = cfg.data.train
    outlier_data_cfg = cfg.data.outlier
    train_data_cfg['pipeline'] = [
        x for x in train_data_cfg.pipeline if x['type'] not in skip_type
    ]
    outlier_data_cfg['pipeline'] = [
        x for x in outlier_data_cfg.pipeline if x['type'] not in skip_type
    ]
    return cfg


def main():
    args = parse_args()


    if args.outlier:
        # OUTLIER
        cfg = retrieve_data_cfg_outlier(args.config, args.skip_type)
        print("\ncfg.data.outlier.items():")
        for k,v in cfg.data.outlier.items():
            print(k,v)
        dataset = build_dataset(cfg.data.outlier)
    else:
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

    for item in dataset:
        filename = os.path.join(args.output_dir,
                                Path(item['filename']).name
                                ) if args.output_dir is not None else None

        gt_masks = item.get('gt_masks', None)
        if gt_masks is not None:
            gt_masks = mask2ndarray(gt_masks)


        #print("\n\n")
        #for k,v in item.items():
        #    print("\n",k,v)
        #    if k == 'ori_shape':
        #        if v[0] > v[1]:
        #            exit()

        print("\n\n")
        print('ori_shape:', item['ori_shape'])
        print('img_shape:', item['img_shape'])
        print('img.shape:', item['img'].shape)

        continue

        #if 'aux_' in item['filename'].split('/')[-1]:
        #    print("\n\n")
        #    for k,v in item.items():
        #        print("\n",k,v)
                
            #print("filename:", item['filename'])
            #print("img:", item['img'].shape)
            #print("img_shape:", item['img_shape'])
            #print("gt_bboxes:", item['gt_bboxes'])
            #print("gt_labels:", item['gt_labels'])
            #exit()

        imshow_det_bboxes(
            item['img'],
            item['gt_bboxes'],
            item['gt_labels'],
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
