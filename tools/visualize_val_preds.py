import argparse
import os
import copy
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import mmcv
from mmcv import Config, DictAction

from mmdet.core.utils import mask2ndarray
from mmdet.core.visualization import imshow_det_bboxes, imshow_gt_det_bboxes
from mmdet.datasets import build_dataset, get_loading_pipeline


def parse_args():
    parser = argparse.ArgumentParser(description='Compare predictions of a variety of models on test images')
    parser.add_argument('config', help='config file path')
    parser.add_argument('preds', help='preds pkl file')
    parser.add_argument('--threshold', type=float, default=0.80)
    parser.add_argument(
        '--skip-type',
        type=str,
        nargs='+',
        default=['Resize', 'RandomFlip', 'MinIoURandomCrop', 'DefaultFormatBundle', 'Pad', 'Normalize', 'Collect'],
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
        default=1,
        help='the interval of show (s)')
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
    args = parser.parse_args()
    return args


def retrieve_data_cfg(config_path, skip_type):
    cfg = Config.fromfile(config_path)
    train_data_cfg = cfg.data.train
    train_data_cfg['pipeline'] = [
        x for x in train_data_cfg.pipeline if x['type'] not in skip_type
    ]
    # Use test images
    train_data_cfg.ann_file = cfg.data.test.ann_file
    train_data_cfg.img_prefix = cfg.data.test.img_prefix
    return cfg




def main():
    args = parse_args()

    mmcv.check_file_exist(args.preds)

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    cfg.data.test.test_mode = True
    if cfg.get('custom_imports', None):
        from mmcv.utils import import_modules_from_strings
        import_modules_from_strings(**cfg['custom_imports'])

    cfg.data.test.pop('samples_per_gpu', 0)
    cfg.data.test.pipeline = get_loading_pipeline(cfg.data.train.pipeline)
    for k,v in cfg.data.test.items():
        print(k,v)
    dataset = build_dataset(cfg.data.test)
    preds = mmcv.load(args.preds)
    print("preds:", len(preds))
    


    # TRAIN
    #cfg = retrieve_data_cfg(args.config, args.skip_type)
    #print("\ncfg.data.train.items():")
    #for k,v in cfg.data.train.items():
    #    print(k,v)

    #preds = mmcv.load(args.preds)
    #print("preds:", len(preds))

    #for i in range(len(preds)):
    #    print(i, len(preds[i][0]))

    #dataset = build_dataset(cfg.data.train)
    #subset_size = 1000
    #dataset = torch.utils.data.random_split(dataset, [subset_size, len(dataset)-subset_size])[0]

    progress_bar = mmcv.ProgressBar(len(dataset))


    #for i, item in enumerate(dataset):
        #filename = os.path.join(args.output_dir,
        #                        Path(item['filename']).name
        #                        ) if args.output_dir is not None else None

        #for k,v in item.items():
        #    print("\n",k,v)

    for i, (result, ) in enumerate(zip(preds)):
        # self.dataset[i] should not call directly
        # because there is a risk of mismatch
        item = dataset.prepare_train_img(i)
        filename = os.path.join(args.output_dir, item['ori_filename']) if args.output_dir is not None else None
        print("\n\nitem:")
        for k,v in item.items():
            print(k,v)
        print("\n\nfilename:", filename)


 
        # Get relevant predictions for this image
        curr_pred_bboxes = preds[i][0]
        filtered_curr_pred_bboxes = curr_pred_bboxes[curr_pred_bboxes[:, -1] >= args.threshold]
        filtered_curr_pred_labels = np.zeros(len(filtered_curr_pred_bboxes), dtype=np.int64)
        print("filtered_curr_pred_bboxes:", filtered_curr_pred_bboxes, filtered_curr_pred_bboxes.shape)
        print("filtered_curr_pred_labels:", filtered_curr_pred_labels, filtered_curr_pred_labels.shape)


        # Plot GT boxes
        thickness = 3
        fig_size = (8, 6)
        img_gt = imshow_det_bboxes(
            item['img'],
            item['gt_bboxes'],
            item['gt_labels'],
            bbox_color=(255,17,0),
            text_color=(255,17,0),
            thickness=thickness,
            fig_size=fig_size,
            show=False,
        )
        # Plot pred boxes
        thickness = 1.5
        fig_size = (8, 6)
        box_color = (255, 255, 0)
        imshow_det_bboxes(
            img_gt,
            filtered_curr_pred_bboxes,
            filtered_curr_pred_labels,
            bbox_color=box_color,
            text_color=box_color,
            thickness=thickness,
            fig_size=fig_size,
            show=(not args.not_show),
            wait_time=args.show_interval,
            out_file=filename,
        )


        if i == 200:
            print("done!")
            exit()


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
