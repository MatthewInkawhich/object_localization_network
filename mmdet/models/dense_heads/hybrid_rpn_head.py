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

    def __init__(self, loss_objectness, objectness_type='Centerness', lambda_cls=0.50, ss=False, lwbr=False, score_scale=1, **kwargs):
        super(HybridOlnRPNHead, self).__init__(loss_objectness, **kwargs)
        self.lambda_cls = lambda_cls
        self.ss = ss
        self.lwbr = lwbr
        self.score_scale=score_scale



    def _get_targets_single(self,
                            flat_anchors,
                            valid_flags,
                            gt_bboxes,
                            gt_bboxes_ignore,
                            gt_labels,
                            gt_scores,
                            img_meta,
                            label_channels=1,
                            unmap_outputs=True):
        """Compute regression and classification targets for anchors in a
        single image.

        Args:
            flat_anchors (Tensor): Multi-level anchors of the image, which are
                concatenated into a single tensor of shape (num_anchors ,4)
            valid_flags (Tensor): Multi level valid flags of the image,
                which are concatenated into a single tensor of
                    shape (num_anchors,).
            gt_bboxes (Tensor): Ground truth bboxes of the image,
                shape (num_gts, 4).
            img_meta (dict): Meta info of the image.
            gt_bboxes_ignore (Tensor): Ground truth bboxes to be
                ignored, shape (num_ignored_gts, 4).
            img_meta (dict): Meta info of the image.
            gt_labels (Tensor): Ground truth labels of each box,
                shape (num_gts,).
            label_channels (int): Channel of label.
            unmap_outputs (bool): Whether to map outputs back to the original
                set of anchors.

        Returns:
            tuple:
                labels_list (list[Tensor]): Labels of each level
                label_weights_list (list[Tensor]): Label weights of each level
                bbox_targets_list (list[Tensor]): BBox targets of each level
                bbox_weights_list (list[Tensor]): BBox weights of each level
                num_total_pos (int): Number of positive samples in all images
                num_total_neg (int): Number of negative samples in all images
        """
#        print("\n\n\n ARGS:")
#        print("\n flat_anchors:", flat_anchors, flat_anchors.shape)
#        print("\n valid_flags:", valid_flags, valid_flags.shape)
#        print("\n gt_bboxes:", gt_bboxes, gt_bboxes.shape)
#        print("\n gt_bboxes_ignore:", gt_bboxes_ignore)
#        print("\n gt_labels:", gt_labels)
#        print("\n gt_scores:", gt_scores)
#        print("\n img_meta:", img_meta)
#        print("\n label_channels:", label_channels)
#        print("\n unmap_outputs:", unmap_outputs)


        inside_flags = anchor_inside_flags(flat_anchors, valid_flags,
                                           img_meta['img_shape'][:2],
                                           self.train_cfg.allowed_border)
        if not inside_flags.any():
            return (None, ) * 7
        # assign gt and sample anchors
        anchors = flat_anchors[inside_flags, :]

        assign_result = self.assigner.assign(
            anchors, gt_bboxes, gt_bboxes_ignore,
            None if self.sampling else gt_labels)
        sampling_result = self.sampler.sample(assign_result, anchors,
                                              gt_bboxes)

        # Assign objectness gt and sample anchors
        objectness_assign_result = self.objectness_assigner.assign(
            anchors, gt_bboxes, gt_bboxes_ignore, None)
        objectness_sampling_result = self.objectness_sampler.sample(
            objectness_assign_result, anchors, gt_bboxes)

        num_valid_anchors = anchors.shape[0]
        bbox_targets = torch.zeros_like(anchors)
        bbox_weights = torch.zeros_like(anchors)
        labels = anchors.new_full((num_valid_anchors, ),
                                  self.num_classes,
                                  dtype=torch.long)
        label_weights = anchors.new_zeros(num_valid_anchors, dtype=torch.float)
        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds
        if len(pos_inds) > 0:
            if not self.reg_decoded_bbox:
                pos_bbox_targets = self.bbox_coder.encode(
                    sampling_result.pos_bboxes, sampling_result.pos_gt_bboxes)
            else:
                pos_bbox_targets = sampling_result.pos_gt_bboxes

            # Sanlity check: left, right, top, bottom distances must be greater
            # than 0.
            valid_targets = torch.min(pos_bbox_targets,-1)[0] > 0
            bbox_targets[pos_inds[valid_targets], :] = (
                pos_bbox_targets[valid_targets])

            # New by Mink
            # If there are gt_scores present, weight bbox loss according to 
            # the corresponding GT's score
            #print("\n pos_bbox_targets:", pos_bbox_targets)
            #print("\n bbox_targets:", bbox_targets, bbox_targets.shape)
            #print("\n pos_inds:", pos_inds)
            #print("\n valid_targets:", valid_targets, valid_targets.shape)
            if self.lwbr:
                assert gt_scores is not None, "Error (Hybrid-RPN): self.lwbr==True but gt_scores is None"
                assert "score_beta" in self.train_cfg, "Error (Hybrid-RPN): gt_scores present but no score_beta in rpn_train_cfg"
                num_gts = sampling_result.num_gts
                pos_assigned_gt_inds = sampling_result.pos_assigned_gt_inds
                #print("num_gts:", num_gts)
                #print("pos_assigned_gt_inds:", pos_assigned_gt_inds)
                for gt_idx in range(num_gts):
                    all_conditions = torch.logical_and(valid_targets, (pos_assigned_gt_inds==gt_idx))
                    bbox_weights[pos_inds[all_conditions], :] = gt_scores[gt_idx] ** self.train_cfg.score_beta
            else: 
                bbox_weights[pos_inds[valid_targets], :] = 1.0

            #print("\n bbox_weights:", bbox_weights, bbox_weights.shape)
            #print("bbox_weights indexes of 1.0:")
            #for i in range(bbox_weights.shape[0]):
            #    if bbox_weights[i][0] > 0:
            #        print("found: ", i, bbox_weights[i])
            #exit()

            if gt_labels is None:
                # Only rpn gives gt_labels as None
                # Foreground is the first class since v2.5.0
                labels[pos_inds] = 0
            else:
                labels[pos_inds] = gt_labels[
                    sampling_result.pos_assigned_gt_inds]


            # Selective (loss) Sampling (SS):    
            # If a CLS head prediction matched target is a PL (has gt_score<1.0),
            # manually ZERO the loss for that prediction
            if self.ss:
                #print("gt_labels:", gt_labels)
                #print("gt_scores:", gt_scores, gt_scores.shape)
                #print("\n\nlabels:", labels, labels.shape)
                #print("label_weights:", label_weights, label_weights.shape)
                #print("pos_inds:", pos_inds, pos_inds.shape)
                #print("neg_inds:", neg_inds, neg_inds.shape)
                #print("num_gts:", sampling_result.num_gts)
                #print("pos_assigned_gt_inds:", sampling_result.pos_assigned_gt_inds)
                # First turn all label weights corresponding to PLs to the target PL's score
                label_weights[pos_inds] = gt_scores[sampling_result.pos_assigned_gt_inds]
                # Now we can easily zero the weights for PLs (the ones > 0 and < 1.0)
                label_weights[label_weights < 1.0] = 0

                if self.train_cfg.pos_weight > 0:
                    label_weights *= self.train_cfg.pos_weight

                #print("\n\nnew label_weights:", label_weights, label_weights.shape, label_weights.sum())
                #for i in range(label_weights.shape[0]):
                #    if label_weights[i] > 0:
                #        print(i, label_weights[i])
                #exit()

            else:
                if self.train_cfg.pos_weight <= 0:
                    label_weights[pos_inds] = 1.0
                else:
                    label_weights[pos_inds] = self.train_cfg.pos_weight

        if len(neg_inds) > 0:
            label_weights[neg_inds] = 1.0


        objectness_targets = anchors.new_zeros(
            num_valid_anchors, dtype=torch.float)
        objectness_weights = anchors.new_zeros(
            num_valid_anchors, dtype=torch.float)
        objectness_pos_inds = objectness_sampling_result.pos_inds
        objectness_neg_inds = objectness_sampling_result.neg_inds
        objectness_pos_neg_inds = torch.cat(
            [objectness_pos_inds, objectness_neg_inds])

        if len(objectness_pos_inds) > 0:
            # Centerness as tartet -- Default
            if self.objectness_type == 'Centerness':
                pos_objectness_bbox_targets = self.bbox_coder.encode(
                    objectness_sampling_result.pos_bboxes, 
                    objectness_sampling_result.pos_gt_bboxes)
                valid_targets = torch.min(pos_objectness_bbox_targets,-1)[0] > 0
                pos_objectness_bbox_targets[valid_targets==False,:] = 0
                top_bottom = pos_objectness_bbox_targets[:,0:2]
                left_right = pos_objectness_bbox_targets[:,2:4]
                pos_objectness_targets = torch.sqrt(
                    (torch.min(top_bottom, -1)[0] / 
                        (torch.max(top_bottom, -1)[0] + 1e-12)) *
                    (torch.min(left_right, -1)[0] / 
                        (torch.max(left_right, -1)[0] + 1e-12)))
            elif self.objectness_type == 'BoxIoU':
                pos_objectness_targets = bbox_overlaps(
                    objectness_sampling_result.pos_bboxes,
                    objectness_sampling_result.pos_gt_bboxes,
                    is_aligned=True)
            else:
                raise ValueError(
                    'objectness_type must be either "Centerness" (Default) or '
                    '"BoxIoU".')

            objectness_targets[objectness_pos_inds] = pos_objectness_targets
            objectness_weights[objectness_pos_inds] = 1.0   

        if len(objectness_neg_inds) > 0: 
            objectness_targets[objectness_neg_inds] = 0.0
            objectness_weights[objectness_neg_inds] = 1.0

        # map up to original set of anchors
        if unmap_outputs:
            num_total_anchors = flat_anchors.size(0)
            labels = unmap(
                labels, num_total_anchors, inside_flags,
                fill=self.num_classes)  # fill bg label
            label_weights = unmap(label_weights, num_total_anchors,
                                  inside_flags)
            bbox_targets = unmap(bbox_targets, num_total_anchors, inside_flags)
            bbox_weights = unmap(bbox_weights, num_total_anchors, inside_flags)

            # objectness targets
            objectness_targets = unmap(
                objectness_targets, num_total_anchors, inside_flags)
            objectness_weights = unmap(
                objectness_weights, num_total_anchors, inside_flags)

        return (labels, label_weights, bbox_targets, bbox_weights, pos_inds,
                neg_inds, sampling_result,
                objectness_targets, objectness_weights, 
                objectness_pos_inds, objectness_neg_inds, objectness_pos_neg_inds,
                objectness_sampling_result)




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

            #print("\n\nRPN...")
            #print("self.lambda_cls:", self.lambda_cls)
            #print("rpn_objectness_scores:", rpn_objectness_scores, rpn_objectness_scores.shape, rpn_objectness_scores.min(), rpn_objectness_scores.max(), rpn_objectness_scores.mean(), rpn_objectness_scores.median())
            #print("rpn_cls_scores:", rpn_cls_scores, rpn_cls_scores.shape, rpn_cls_scores.min(), rpn_cls_scores.max(), rpn_cls_scores.mean(), rpn_cls_scores.median())

            scores = self.lambda_cls*(rpn_cls_scores**self.score_scale) + (1-self.lambda_cls)*(rpn_objectness_scores**self.score_scale)

            #print("scores:", scores, scores.shape, scores.min(), scores.max(), scores.mean(), scores.median())
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

        
