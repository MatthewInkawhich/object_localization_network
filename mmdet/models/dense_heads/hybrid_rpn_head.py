# Mink
"""This file contains code to build Hybrid OLN/RPN

Reference:
    "Learning Open-World Object Proposals without Learning to Classify",
        Aug 2021. https://arxiv.org/abs/2108.06753
        Dahun Kim, Tsung-Yi Lin, Anelia Angelova, In So Kweon and Weicheng Kuo
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import normal_init
from mmcv.ops import batched_nms
from mmcv.runner import force_fp32

from mmdet.core import (anchor_inside_flags, build_anchor_generator,
                        build_assigner, build_bbox_coder, build_sampler,
                        images_to_levels, multi_apply, multiclass_nms, unmap)
from mmdet.core.bbox import bbox_overlaps
from ..builder import HEADS, build_loss
from .oln_rpn_head import OlnRPNHead


@HEADS.register_module()
class HybridOlnRPNHead(OlnRPNHead):
    """
        Hybrid OLN+RPN head
    """

    def __init__(self, loss_objectness, objectness_type='Centerness', lambda_cls=0.50, **kwargs):
        super(HybridOlnRPNHead, self).__init__(loss_objectness, **kwargs)
        self.lambda_cls = lambda_cls

    def _get_bboxes_single(self,
                           cls_scores,
                           bbox_preds,
                           objectness_scores,
                           mlvl_anchors,
                           img_shape,
                           scale_factor,
                           cfg,
                           rescale=False):
        """Transform outputs for a single batch item into bbox predictions.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level
                Has shape (num_anchors * num_classes, H, W).
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level with shape (num_anchors * 4, H, W).
            objectness_score_list (list[Tensor]): Box objectness scorees for
                each anchor point with shape (N, num_anchors, H, W)
            mlvl_anchors (list[Tensor]): Box reference for each scale level
                with shape (num_total_anchors, 4).
            img_shape (tuple[int]): Shape of the input image,
                (height, width, 3).
            scale_factor (ndarray): Scale factor of the image arange as
                (w_scale, h_scale, w_scale, h_scale).
            cfg (mmcv.Config): Test / postprocessing configuration,
                if None, test_cfg would be used.
            rescale (bool): If True, return boxes in original image space.

        Returns:
            Tensor: Labeled boxes in shape (n, 5), where the first 4 columns
                are bounding box positions (tl_x, tl_y, br_x, br_y) and the
                5-th column is a score between 0 and 1.
        """
        cfg = self.test_cfg if cfg is None else cfg
        # bboxes from different level should be independent during NMS,
        # level_ids are used as labels for batched NMS to separate them
        level_ids = []
        mlvl_scores = []
        mlvl_bbox_preds = []
        mlvl_valid_anchors = []
        for idx in range(len(cls_scores)):
            rpn_cls_score = cls_scores[idx]
            rpn_bbox_pred = bbox_preds[idx]
            # <
            rpn_objectness_score = objectness_scores[idx]
            # >

            assert rpn_cls_score.size()[-2:] == rpn_bbox_pred.size()[-2:]
            rpn_cls_score = rpn_cls_score.permute(1, 2, 0)
            
            assert self.use_sigmoid_cls, 'use_sigmoid_cls must be True.'
            rpn_cls_score = rpn_cls_score.reshape(-1)
            rpn_cls_scores = rpn_cls_score.sigmoid()

            #print("rpn_objectness_score (BEFORE):", rpn_objectness_score, rpn_objectness_score.shape)
            rpn_objectness_score = rpn_objectness_score.permute(
                1, 2, 0).reshape(-1)
            rpn_objectness_scores = rpn_objectness_score.sigmoid()
            
            ########################################################################
            # scores = lambda_cls*cls_scores + (1-lambda_cls)*objectness_scores

            #print("rpn_objectness_scores (AFTER):", rpn_objectness_scores, rpn_objectness_scores.shape)
            #print("rpn_cls_scores:", rpn_cls_scores, rpn_cls_scores.shape)
            #print("self.lambda_cls:", self.lambda_cls)

            scores = self.lambda_cls*rpn_cls_scores + (1 - self.lambda_cls)*rpn_objectness_scores

            #print("scores:", scores, scores.shape)
            #exit()

            ########################################################################
            
            rpn_bbox_pred = rpn_bbox_pred.permute(1, 2, 0).reshape(-1, 4)
            anchors = mlvl_anchors[idx]
            if cfg.nms_pre > 0 and scores.shape[0] > cfg.nms_pre:
                ranked_scores, rank_inds = scores.sort(descending=True)
                topk_inds = rank_inds[:cfg.nms_pre]
                scores = ranked_scores[:cfg.nms_pre]
                rpn_bbox_pred = rpn_bbox_pred[topk_inds, :]
                anchors = anchors[topk_inds, :]
            mlvl_scores.append(scores)
            mlvl_bbox_preds.append(rpn_bbox_pred)
            mlvl_valid_anchors.append(anchors)
            level_ids.append(
                scores.new_full((scores.size(0), ), idx, dtype=torch.long))

        scores = torch.cat(mlvl_scores)
        anchors = torch.cat(mlvl_valid_anchors)
        rpn_bbox_pred = torch.cat(mlvl_bbox_preds)
        proposals = self.bbox_coder.decode(
            anchors, rpn_bbox_pred, max_shape=img_shape)
        ids = torch.cat(level_ids)

        if cfg.min_bbox_size > 0:
            w = proposals[:, 2] - proposals[:, 0]
            h = proposals[:, 3] - proposals[:, 1]
            valid_inds = torch.nonzero(
                (w >= cfg.min_bbox_size)
                & (h >= cfg.min_bbox_size),
                as_tuple=False).squeeze()
            if valid_inds.sum().item() != len(proposals):
                proposals = proposals[valid_inds, :]
                scores = scores[valid_inds]
                ids = ids[valid_inds]

        nms_cfg = dict(type='nms', iou_threshold=cfg.nms_thr)

        # No NMS:
        dets = torch.cat([proposals, scores.unsqueeze(1)], 1)
        
        return dets

        
