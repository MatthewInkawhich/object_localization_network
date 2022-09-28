"""This file contains code to build box-scoring head of OLN-Box head.

Reference:
    "Learning Open-World Object Proposals without Learning to Classify",
        Aug 2021. https://arxiv.org/abs/2108.06753
        Dahun Kim, Tsung-Yi Lin, Anelia Angelova, In So Kweon and Weicheng Kuo
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

#import matplotlib.pyplot as plt

from mmcv.cnn import ConvModule
from mmcv.runner import force_fp32

from mmdet.core import multi_apply, multiclass_nms, build_bbox_coder
from mmdet.core.bbox import bbox_overlaps
from mmdet.models.builder import HEADS, build_loss
from mmdet.models.losses import accuracy

from .bbox_head import BBoxHead
from .convfc_bbox_head import ConvFCBBoxHead


@HEADS.register_module()
class ConvFCBBoxScoreHead(ConvFCBBoxHead):
    r"""More general bbox scoring head, to construct the OLN-Box head. It
    consists of shared conv and fc layers and three separated branches as below.

    .. code-block:: none

                                    /-> cls fcs -> cls
        shared convs -> shared fcs
                                    \-> reg fcs -> reg

                                    \-> bbox-scoring fcs -> bbox-score
    """  # noqa: W605

    def __init__(self, 
                 with_bbox_score=True, 
                 bbox_score_type='BoxIoU',
                 loss_bbox_score=dict(type='L1Loss', loss_weight=1.0),
                 **kwargs):
        super(ConvFCBBoxScoreHead, self).__init__(**kwargs)
        self.with_bbox_score = with_bbox_score
        if self.with_bbox_score:
            self.fc_bbox_score = nn.Linear(self.cls_last_dim, 1)

        self.loss_bbox_score = build_loss(loss_bbox_score)
        self.bbox_score_type = bbox_score_type
        # Mink
        if loss_bbox_score['type'] == 'QualityOnlyFocalLoss':
            self.qofl = True
        else:
            self.qofl = False

        self.with_class_score = self.loss_cls.loss_weight > 0.0
        self.with_bbox_loc_score = self.loss_bbox_score.loss_weight > 0.0

    def init_weights(self):
        super(ConvFCBBoxScoreHead, self).init_weights()
        if self.with_bbox_score:
            nn.init.normal_(self.fc_bbox_score.weight, 0, 0.01)
            nn.init.constant_(self.fc_bbox_score.bias, 0)

    def forward(self, x):
        # shared part
        if self.num_shared_convs > 0:
            for conv in self.shared_convs:
                x = conv(x)

        if self.num_shared_fcs > 0:
            if self.with_avg_pool:
                x = self.avg_pool(x)

            x = x.flatten(1)

            for fc in self.shared_fcs:
                x = self.relu(fc(x))
        # separate branches
        x_cls = x
        x_reg = x
        x_bbox_score = x

        cls_score = self.fc_cls(x_cls) if self.with_cls else None
        bbox_pred = self.fc_reg(x_reg) if self.with_reg else None
        bbox_score = (self.fc_bbox_score(x_bbox_score)
                      if self.with_bbox_score else None)

        return cls_score, bbox_pred, bbox_score

    def _get_target_single(self, pos_bboxes, neg_bboxes, pos_gt_bboxes,
                           pos_gt_labels, gt_scores, sampling_result, cfg):
        num_pos = pos_bboxes.size(0)
        num_neg = neg_bboxes.size(0)
        num_samples = num_pos + num_neg

        # original implementation uses new_zeros since BG are set to be 0
        # now use empty & fill because BG cat_id = num_classes,
        # FG cat_id = [0, num_classes-1]
        labels = pos_bboxes.new_full((num_samples, ),
                                     self.num_classes,
                                     dtype=torch.long)
        label_weights = pos_bboxes.new_zeros(num_samples)
        bbox_targets = pos_bboxes.new_zeros(num_samples, 4)
        bbox_weights = pos_bboxes.new_zeros(num_samples, 4)
        bbox_score_targets = pos_bboxes.new_zeros(num_samples)
        bbox_score_weights = pos_bboxes.new_zeros(num_samples)

        if num_pos > 0:
            labels[:num_pos] = pos_gt_labels
            pos_weight = 1.0 if cfg.pos_weight <= 0 else cfg.pos_weight
            label_weights[:num_pos] = pos_weight

            if not self.reg_decoded_bbox:
                pos_bbox_targets = self.bbox_coder.encode(
                    pos_bboxes, pos_gt_bboxes)
            else:
                # When the regression loss (e.g. `IouLoss`, `GIouLoss`)
                # is applied directly on the decoded bounding boxes, both
                # the predicted boxes and regression targets should be with
                # absolute coordinate format.
                pos_bbox_targets = pos_gt_bboxes
            bbox_targets[:num_pos, :] = pos_bbox_targets

            # New by Mink
            # If there are gt_scores present, weight bbox loss according to 
            # the corresponding GT's score
            #print("\n\nCurr sampling_result:", sampling_result)
            #print("\n num_pos:", num_pos)
            if gt_scores is None:
                bbox_weights[:num_pos, :] = 1
            else:
                assert "score_beta" in cfg, "Error (OLN-ROI): gt_scores present but no score_beta in rcnn_train_cfg"
                #print("\n gt_scores:", gt_scores)
                num_gts = sampling_result.num_gts
                pos_assigned_gt_inds = sampling_result.pos_assigned_gt_inds
                pos_bbox_weights = pos_bboxes.new_zeros(num_pos, 4)
                for gt_idx in range(num_gts):
                    pos_bbox_weights[(pos_assigned_gt_inds == gt_idx), :] = gt_scores[gt_idx] ** cfg.score_beta
                #print("\n pos_bbox_weights:", pos_bbox_weights, pos_bbox_weights.shape)
                bbox_weights[:num_pos, :] = pos_bbox_weights

            #print("\n bbox_weights:", bbox_weights.shape)
            #for i in range(bbox_weights.shape[0]):
            #    print(i, bbox_weights[i])
            #exit()
            
            # Bbox-IoU as target
            if self.bbox_score_type == 'BoxIoU':
                pos_bbox_score_targets = bbox_overlaps(
                    pos_bboxes, pos_gt_bboxes, is_aligned=True)
            # Centerness as target
            elif self.bbox_score_type == 'Centerness':
                tblr_bbox_coder = build_bbox_coder(
                    dict(type='TBLRBBoxCoder', normalizer=1.0))
                pos_center_bbox_targets = tblr_bbox_coder.encode(
                    pos_bboxes, pos_gt_bboxes)
                valid_targets = torch.min(pos_center_bbox_targets,-1)[0] > 0
                pos_center_bbox_targets[valid_targets==False,:] = 0
                top_bottom = pos_center_bbox_targets[:,0:2]
                left_right = pos_center_bbox_targets[:,2:4]
                pos_bbox_score_targets = torch.sqrt(
                    (torch.min(top_bottom, -1)[0] / 
                        (torch.max(top_bottom, -1)[0] + 1e-12)) *
                    (torch.min(left_right, -1)[0] / 
                        (torch.max(left_right, -1)[0] + 1e-12)))
            else:
                raise ValueError(
                    'bbox_score_type must be either "BoxIoU" (Default) or \
                    "Centerness".')

            bbox_score_targets[:num_pos] = pos_bbox_score_targets
            bbox_score_weights[:num_pos] = 1

        if num_neg > 0:
            label_weights[-num_neg:] = 1.0

        return (labels, label_weights, bbox_targets, bbox_weights,
                bbox_score_targets, bbox_score_weights)

    def get_targets(self,
                    sampling_results,
                    gt_bboxes,
                    gt_labels,
                    gt_scores, # New
                    rcnn_train_cfg,
                    concat=True,
                    class_agnostic=False):

        pos_bboxes_list = [res.pos_bboxes for res in sampling_results]
        neg_bboxes_list = [res.neg_bboxes for res in sampling_results]
        pos_gt_bboxes_list = [res.pos_gt_bboxes for res in sampling_results]
        pos_gt_labels_list = [res.pos_gt_labels for res in sampling_results]
        if gt_scores is None:
            gt_scores = [None for _ in range(len(sampling_results))]
        (labels, label_weights, bbox_targets, bbox_weights, 
         bbox_score_targets, bbox_score_weights) = multi_apply(
            self._get_target_single,
            pos_bboxes_list,
            neg_bboxes_list,
            pos_gt_bboxes_list,
            pos_gt_labels_list,
            gt_scores,
            sampling_results,
            cfg=rcnn_train_cfg)

        if concat:
            labels = torch.cat(labels, 0)
            label_weights = torch.cat(label_weights, 0)
            bbox_targets = torch.cat(bbox_targets, 0)
            bbox_weights = torch.cat(bbox_weights, 0)
            bbox_score_targets = torch.cat(bbox_score_targets, 0)
            bbox_score_weights = torch.cat(bbox_score_weights, 0)

        return (labels, label_weights, bbox_targets, bbox_weights,
                bbox_score_targets, bbox_score_weights)

    @force_fp32(apply_to=('cls_score', 'bbox_pred', 'bbox_score'))
    def loss(self,
             cls_score,
             bbox_pred,
             bbox_score,
             rois,
             labels,
             label_weights,
             bbox_targets,
             bbox_weights,
             bbox_score_targets,
             bbox_score_weights,
             reduction_override=None):
        losses = dict()
        if cls_score is not None:
            avg_factor = max(torch.sum(label_weights > 0).float().item(), 1.)
            if cls_score.numel() > 0:
                #print("cls_score:", cls_score, cls_score.shape)
                #print("labels:", labels, labels.shape)
                #print("label_weights:", label_weights, label_weights.shape)
                losses['loss_cls'] = self.loss_cls(
                    cls_score,
                    labels,
                    label_weights,
                    avg_factor=avg_factor,
                    reduction_override=reduction_override)
                #print("losses[loss_cls]:", losses['loss_cls'])
                #exit()

                losses['acc'] = accuracy(cls_score, labels)
        if bbox_pred is not None:
            bg_class_ind = self.num_classes
            # 0~self.num_classes-1 are FG, self.num_classes is BG
            pos_inds = (labels >= 0) & (labels < bg_class_ind)
            # do not perform bounding box regression for BG anymore.
            if pos_inds.any():
                if self.reg_decoded_bbox:
                    # When the regression loss (e.g. `IouLoss`,
                    # `GIouLoss`, `DIouLoss`) is applied directly on
                    # the decoded bounding boxes, it decodes the
                    # already encoded coordinates to absolute format.
                    bbox_pred = self.bbox_coder.decode(rois[:, 1:], bbox_pred)
                if self.reg_class_agnostic:
                    pos_bbox_pred = bbox_pred.view(
                        bbox_pred.size(0), 4)[pos_inds.type(torch.bool)]
                else:
                    pos_bbox_pred = bbox_pred.view(
                        bbox_pred.size(0), -1,
                        4)[pos_inds.type(torch.bool),
                           labels[pos_inds.type(torch.bool)]]

                losses['loss_bbox'] = self.loss_bbox(
                    pos_bbox_pred,
                    bbox_targets[pos_inds.type(torch.bool)],
                    bbox_weights[pos_inds.type(torch.bool)],
                    avg_factor=bbox_targets.size(0),
                    reduction_override=reduction_override)
            else:
                losses['loss_bbox'] = bbox_pred[pos_inds].sum()

        if bbox_score is not None:
            if bbox_score.numel() > 0:
                if self.qofl:
                    losses['loss_bbox_score'] = self.loss_bbox_score(
                        bbox_score.squeeze(-1),  # We apply sigmoid in BCEwithlogits
                        bbox_score_targets,
                        bbox_score_weights,
                        avg_factor=bbox_score_targets.size(0),
                        reduction_override=reduction_override)
                else:
                    losses['loss_bbox_score'] = self.loss_bbox_score(
                        bbox_score.squeeze(-1).sigmoid(),
                        bbox_score_targets,
                        bbox_score_weights,
                        avg_factor=bbox_score_targets.size(0),
                        reduction_override=reduction_override)
        return losses

    @force_fp32(apply_to=('cls_score', 'bbox_pred'))
    def get_bboxes(self,
                   rois,
                   cls_score,
                   bbox_pred,
                   bbox_score,
                   rpn_score,
                   img_shape,
                   scale_factor,
                   rescale=False,
                   cfg=None):
        if isinstance(cls_score, list):
            cls_score = sum(cls_score) / float(len(cls_score))
        # cls_score is not used.
        # scores = F.softmax(
        #     cls_score, dim=1) if cls_score is not None else None
        
        if bbox_pred is not None:
            bboxes = self.bbox_coder.decode(
                rois[:, 1:], bbox_pred, max_shape=img_shape)
        else:
            bboxes = rois[:, 1:].clone()
            if img_shape is not None:
                bboxes[:, [0, 2]].clamp_(min=0, max=img_shape[1])
                bboxes[:, [1, 3]].clamp_(min=0, max=img_shape[0])

        if rescale and bboxes.size(0) > 0:
            if isinstance(scale_factor, float):
                bboxes /= scale_factor
            else:
                scale_factor = bboxes.new_tensor(scale_factor)
                bboxes = (bboxes.view(bboxes.size(0), -1, 4) /
                          scale_factor).view(bboxes.size()[0], -1)

        # The objectness score of a region is computed as a geometric mean of
        # the estimated localization quality scores of OLN-RPN and OLN-Box
        # heads.
        scores = torch.sqrt(rpn_score * bbox_score.sigmoid())

        # Concat dummy zero-scores for the background class.
        scores = torch.cat([scores, torch.zeros_like(scores)], dim=-1)

        if cfg is None:
            return bboxes, scores
        else:
            det_bboxes, det_labels = multiclass_nms(bboxes, 
                                                    scores,
                                                    cfg.score_thr, cfg.nms,
                                                    cfg.max_per_img)

            return det_bboxes, det_labels



@HEADS.register_module()
class ConvFCBBoxHybridHead(ConvFCBBoxScoreHead):
    """
        Hybrid between Softmax bbox head and OLN bbox head
    """  
    def __init__(self, lambda_cls=0.5, ss=False, lwbr=False, score_scale=1, **kwargs):
        super(ConvFCBBoxHybridHead, self).__init__(**kwargs)
        self.lambda_cls = lambda_cls
        self.ss = ss
        self.lwbr = lwbr
        self.score_scale = score_scale



    def _get_target_single(self, pos_bboxes, neg_bboxes, pos_gt_bboxes,
                           pos_gt_labels, gt_scores, sampling_result, cfg):
        num_pos = pos_bboxes.size(0)
        num_neg = neg_bboxes.size(0)
        num_samples = num_pos + num_neg

        # original implementation uses new_zeros since BG are set to be 0
        # now use empty & fill because BG cat_id = num_classes,
        # FG cat_id = [0, num_classes-1]
        labels = pos_bboxes.new_full((num_samples, ),
                                     self.num_classes,
                                     dtype=torch.long)
        label_weights = pos_bboxes.new_zeros(num_samples)
        bbox_targets = pos_bboxes.new_zeros(num_samples, 4)
        bbox_weights = pos_bboxes.new_zeros(num_samples, 4)
        bbox_score_targets = pos_bboxes.new_zeros(num_samples)
        bbox_score_weights = pos_bboxes.new_zeros(num_samples)

        if num_pos > 0:
            labels[:num_pos] = pos_gt_labels

            if self.ss:
                #print("\n\ngt_scores:", gt_scores, gt_scores.shape)
                #print("num_gts:", sampling_result.num_gts)
                #print("sampling_result.pos_assigned_gt_inds:", sampling_result.pos_assigned_gt_inds, sampling_result.pos_assigned_gt_inds.shape)
                #print("sampling_result.pos_inds:", sampling_result.pos_inds, sampling_result.pos_inds.shape)
                #print("num_pos:", num_pos)
                #print("pos_gt_labels:", pos_gt_labels, pos_gt_labels.shape)
                #print("labels:", labels, labels.shape)

                # First turn all label weights corresponding to PLs to the target PL's score
                label_weights[:num_pos] = gt_scores[sampling_result.pos_assigned_gt_inds]
                #print("\nlabel_weights:", label_weights, label_weights.shape)
                # Now we can easily zero the weights for PLs (the ones > 0 and < 1.0)
                label_weights[label_weights < 1.0] = 0
                if cfg.pos_weight > 0:
                    label_weights *= cfg.pos_weight
                #print("\nlabel_weights:", label_weights, label_weights.shape)
                #exit()

            else:
                pos_weight = 1.0 if cfg.pos_weight <= 0 else cfg.pos_weight
                label_weights[:num_pos] = pos_weight


            if not self.reg_decoded_bbox:
                pos_bbox_targets = self.bbox_coder.encode(
                    pos_bboxes, pos_gt_bboxes)
            else:
                # When the regression loss (e.g. `IouLoss`, `GIouLoss`)
                # is applied directly on the decoded bounding boxes, both
                # the predicted boxes and regression targets should be with
                # absolute coordinate format.
                pos_bbox_targets = pos_gt_bboxes
            bbox_targets[:num_pos, :] = pos_bbox_targets

            # New by Mink
            # If there are gt_scores present, weight bbox loss according to 
            # the corresponding GT's score
            #print("\n\nCurr sampling_result:", sampling_result)
            #print("\n num_pos:", num_pos)
            if self.lwbr:
                assert gt_scores is not None, "Error (Hybrid-RoI): self.lwbr==True but gt_scores is None"
                assert "score_beta" in cfg, "Error (Hybrid-RoI): gt_scores present but no score_beta in rcnn_train_cfg"
                #print("\n gt_scores:", gt_scores)
                num_gts = sampling_result.num_gts
                pos_assigned_gt_inds = sampling_result.pos_assigned_gt_inds
                pos_bbox_weights = pos_bboxes.new_zeros(num_pos, 4)
                for gt_idx in range(num_gts):
                    pos_bbox_weights[(pos_assigned_gt_inds == gt_idx), :] = gt_scores[gt_idx] ** cfg.score_beta
                #print("\n pos_bbox_weights:", pos_bbox_weights, pos_bbox_weights.shape)
                bbox_weights[:num_pos, :] = pos_bbox_weights
            else:
                bbox_weights[:num_pos, :] = 1

            #print("\n bbox_weights:", bbox_weights.shape)
            #for i in range(bbox_weights.shape[0]):
            #    print(i, bbox_weights[i])
            #exit()
            
            # Bbox-IoU as target
            if self.bbox_score_type == 'BoxIoU':
                pos_bbox_score_targets = bbox_overlaps(
                    pos_bboxes, pos_gt_bboxes, is_aligned=True)
            # Centerness as target
            elif self.bbox_score_type == 'Centerness':
                tblr_bbox_coder = build_bbox_coder(
                    dict(type='TBLRBBoxCoder', normalizer=1.0))
                pos_center_bbox_targets = tblr_bbox_coder.encode(
                    pos_bboxes, pos_gt_bboxes)
                valid_targets = torch.min(pos_center_bbox_targets,-1)[0] > 0
                pos_center_bbox_targets[valid_targets==False,:] = 0
                top_bottom = pos_center_bbox_targets[:,0:2]
                left_right = pos_center_bbox_targets[:,2:4]
                pos_bbox_score_targets = torch.sqrt(
                    (torch.min(top_bottom, -1)[0] / 
                        (torch.max(top_bottom, -1)[0] + 1e-12)) *
                    (torch.min(left_right, -1)[0] / 
                        (torch.max(left_right, -1)[0] + 1e-12)))
            else:
                raise ValueError(
                    'bbox_score_type must be either "BoxIoU" (Default) or \
                    "Centerness".')

            bbox_score_targets[:num_pos] = pos_bbox_score_targets
            bbox_score_weights[:num_pos] = 1

        if num_neg > 0:
            label_weights[-num_neg:] = 1.0

        return (labels, label_weights, bbox_targets, bbox_weights,
                bbox_score_targets, bbox_score_weights)



    @force_fp32(apply_to=('cls_score', 'bbox_pred'))
    def get_bboxes(self,
                   rois,
                   cls_score,
                   bbox_pred,
                   bbox_score,
                   rpn_score,
                   img_shape,
                   scale_factor,
                   rescale=False,
                   cfg=None):
        if isinstance(cls_score, list):
            cls_score = sum(cls_score) / float(len(cls_score))
        # cls_score is not used.
        cls_scores = F.softmax(cls_score, dim=1) if cls_score is not None else None
        #print("\n\ncfg:", cfg)
        #print("rois:", rois, rois.shape)
        #print("cls_scores:", cls_scores, cls_scores.shape)
        #print("bbox_pred:", bbox_pred, bbox_pred.shape)
        #print("bbox_score:", bbox_score, bbox_score.shape)
        #print("rpn_score:", rpn_score, rpn_score.shape)
        #print("img_shape:", img_shape)
        #print("scale_factor:", scale_factor)
        #print("rescale:", rescale)
        
        if bbox_pred is not None:
            bboxes = self.bbox_coder.decode(
                rois[:, 1:], bbox_pred, max_shape=img_shape)
        else:
            bboxes = rois[:, 1:].clone()
            if img_shape is not None:
                bboxes[:, [0, 2]].clamp_(min=0, max=img_shape[1])
                bboxes[:, [1, 3]].clamp_(min=0, max=img_shape[0])

        if rescale and bboxes.size(0) > 0:
            if isinstance(scale_factor, float):
                bboxes /= scale_factor
            else:
                scale_factor = bboxes.new_tensor(scale_factor)
                bboxes = (bboxes.view(bboxes.size(0), -1, 4) /
                          scale_factor).view(bboxes.size()[0], -1)

        ############################################################################
        # The objectness score of a region is computed as a geometric mean of
        # the estimated localization quality scores of OLN-RPN and OLN-Box
        # heads.
        pos_cls_scores = cls_scores[:, :-1]
        objectness_scores = torch.sqrt(rpn_score * bbox_score.sigmoid())

        # Analyze
        #print("\n\nBBox Head...")
        #print("self.lambda_cls:", self.lambda_cls)
        #print("pos_cls_scores:", pos_cls_scores, pos_cls_scores.shape, pos_cls_scores.min(), pos_cls_scores.max(), pos_cls_scores.mean(), pos_cls_scores.median())
        #print("objectness_scores:", objectness_scores, objectness_scores.shape, objectness_scores.min(), objectness_scores.max(), objectness_scores.mean(), objectness_scores.median())

        # Final scores is the weighted interpolation between cls_scores and objectness_scores
        #print("self.lambda_cls:", self.lambda_cls)
        scores = self.lambda_cls*(pos_cls_scores**self.score_scale) + (1-self.lambda_cls)*(objectness_scores**self.score_scale)


        #print("scores:", scores, scores.shape, scores.min(), scores.max(), scores.mean(), scores.median())
        #exit()

        # Concat dummy zero-scores for the background class.
        scores = torch.cat([scores, torch.zeros_like(scores)], dim=-1)
        #print("scores:", scores, scores.shape)
        #exit()
        ############################################################################

        if cfg is None:
            return bboxes, scores
        else:
            det_bboxes, det_labels = multiclass_nms(bboxes, 
                                                    scores,
                                                    cfg.score_thr, cfg.nms,
                                                    cfg.max_per_img)

            return det_bboxes, det_labels


@HEADS.register_module()
class Shared2FCBBoxScoreHead(ConvFCBBoxScoreHead):
    def __init__(self, fc_out_channels=1024, *args, **kwargs):
        super(Shared2FCBBoxScoreHead, self).__init__(
            num_shared_convs=0,
            num_shared_fcs=2,
            num_cls_convs=0,
            num_cls_fcs=0,
            num_reg_convs=0,
            num_reg_fcs=0,
            fc_out_channels=fc_out_channels,
            *args,
            **kwargs)

@HEADS.register_module()
class Shared2FCBBoxHybridHead(ConvFCBBoxHybridHead):
    def __init__(self, fc_out_channels=1024, *args, **kwargs):
        super(Shared2FCBBoxHybridHead, self).__init__(
            num_shared_convs=0,
            num_shared_fcs=2,
            num_cls_convs=0,
            num_cls_fcs=0,
            num_reg_convs=0,
            num_reg_fcs=0,
            fc_out_channels=fc_out_channels,
            *args,
            **kwargs)
