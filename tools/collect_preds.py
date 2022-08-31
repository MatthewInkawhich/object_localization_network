# MatthewInkawhich

import argparse
import os
import warnings

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


def parse_args():
    parser = argparse.ArgumentParser(
        description='Generate pseudolabels')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('score_thr', type=float, help='score threshold')
    parser.add_argument('round', type=int, help='self-training round')
    parser.add_argument(
        '--auxiliary',
        action='store_true',
        help='Whether to use auxiliary set or training set')
    parser.add_argument(
        '--combined',
        action='store_true',
        help='Whether to use combined set')
    parser.add_argument(
        '--val',
        action='store_true',
        help='Whether to collect preds on the val set')
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
    assert not (args.auxiliary and args.combined), "cannot set both auxiliary and combined flags"
    assert not (args.auxiliary and args.val), "cannot set both auxiliary and val flags"
    assert not (args.combined and args.val), "cannot set both combined and val flags"

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

    # Build the dataloader
    # Remember, we want to run a "test" on the training data
    if args.auxiliary:
        cfg.data.test.ann_file = 'data/imagenet/empty_annotations.json'
        cfg.data.test.img_prefix = 'data/imagenet/images/'
    elif args.combined:
        cfg.data.test.ann_file = 'data/coco_imagenet/annotations.json'
        cfg.data.test.img_prefix = 'data/coco_imagenet/images/'
    elif args.val:
        pass
    else:
        if "ShipRSImageNet" in cfg.data.test.ann_file:
            cfg.data.test.ann_file = cfg.data.test.ann_file.replace("val", "train")
        else:
            cfg.data.test.ann_file = cfg.data.test.ann_file.replace("val2017", "train2017")
            cfg.data.test.img_prefix = cfg.data.test.img_prefix.replace("val2017", "train2017")

    dataset = build_dataset(cfg.data.test)
    if rank == 0:
        print("args.score_thr:", args.score_thr)
        print("args.round:", args.round)
        print("args.auxiliary:", args.auxiliary)
        print("args.combined:", args.combined)
        print("args.val:", args.val)
        print("cfg.data.test:", cfg.data.test)
        print("dataset:", len(dataset))
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=samples_per_gpu,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=distributed,
        shuffle=False)

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

    # Run inference on all images
    if not distributed:
        model = MMDataParallel(model, device_ids=[0])
        if args.val:
            outputs = single_gpu_test(model, data_loader, show=False, out_dir=None, show_score_thr=0.3)
        else:
            outputs = single_gpu_collect_preds(model, data_loader, score_thr=args.score_thr)
    else:
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False)
        if args.val:
            outputs = multi_gpu_test(model, data_loader, args.tmpdir, args.gpu_collect)
        else:
            outputs = multi_gpu_collect_preds(model, data_loader, args.score_thr, args.tmpdir, args.gpu_collect)

    # tmp
    #if rank == 0:
    #    print("outputs:", outputs)
    #    print("len(outputs):", len(outputs))
    #exit()

    # Output results to json
    if rank == 0:
        if args.auxiliary:
            jsonfile_prefix = os.path.join(args.checkpoint.replace(args.checkpoint.split('/')[-1], ""), f"auxiliary_preds_round{args.round}")
        elif args.combined:
            jsonfile_prefix = os.path.join(args.checkpoint.replace(args.checkpoint.split('/')[-1], ""), f"combined_preds_round{args.round}")
        elif args.val:
            outfile = os.path.join(args.checkpoint.replace(args.checkpoint.split('/')[-1], ""), "val_preds.pkl")
            mmcv.dump(outputs, outfile)
            print("\noutput dumped to:", outfile)
            return
        else:
            jsonfile_prefix = os.path.join(args.checkpoint.replace(args.checkpoint.split('/')[-1], ""), f"preds_round{args.round}")
        print("\njsonfile_prefix:", jsonfile_prefix)
        dataset.format_results(outputs, jsonfile_prefix=jsonfile_prefix)


if __name__ == '__main__':
    main()
