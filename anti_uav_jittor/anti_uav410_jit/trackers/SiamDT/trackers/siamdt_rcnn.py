import mmcv
import numpy as np
import pycocotools.mask as maskUtils
import jittor as jt
import jittor.nn as nn

from mmdet.models.builder import DETECTORS
from mmdet.models.detectors.two_stage import TwoStageDetector

from mmcv.runner import auto_fp16
from mmdet.core import get_classes, \
    bbox2result, bbox2roi, build_assigner, build_sampler

import copy
from .similarity_encoders import RPN_Similarity_Learning, RCNN_Similarity_Learning

__all__ = ['SiamDTRCNN']


@DETECTORS.register_module()
class SiamDTRCNN(TwoStageDetector):

    def __init__(self,
                 backbone,
                 neck=None,
                 rpn_head=None,
                 roi_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(SiamDTRCNN, self).__init__(
            backbone=backbone,
            neck=neck,
            rpn_head=rpn_head,
            roi_head=roi_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained)
        # build modulators
        self.rpn_similarity_learning = RPN_Similarity_Learning()
        self.rcnn_similarity_learning = RCNN_Similarity_Learning()
        # initialize weights
        self.rpn_similarity_learning.init_weights()
        self.rcnn_similarity_learning.init_weights()

    @auto_fp16(apply_to=('img_z', 'img_x'))
    def forward(self,
                img_z,
                img_x,
                img_meta_z,
                img_meta_x,
                return_loss=True,
                **kwargs):
        if return_loss:
            return self.forward_train(
                img_z, img_x, img_meta_z, img_meta_x, **kwargs)
        else:
            return self.forward_test(
                img_z, img_x, img_meta_z, img_meta_x, **kwargs)

    def forward_dummy(self, *args, **kwargs):
        raise NotImplementedError(
            'forward_dummy is not implemented for SiamDTRCNN')

    def forward_train(self,
                      img_z,
                      img_x,
                      img_meta_z,
                      img_meta_x,
                      gt_bboxes_z,
                      gt_bboxes_x,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      proposals=None):

        z = self.extract_feat(img_z)
        x = self.extract_feat(img_x)

        # common parameters
        proposal_cfg = self.train_cfg.get(
            'rpn_proposal', self.test_cfg.rpn)
        bbox_assigner = build_assigner(
            self.train_cfg.rcnn.assigner)
        bbox_sampler = build_sampler(
            self.train_cfg.rcnn.sampler, context=self)

        losses = dict()

        # get template
        rois_z = bbox2roi(gt_bboxes_z)
        bbox_feats_z = self.roi_head.bbox_roi_extractor(
            z[:self.roi_head.bbox_roi_extractor.num_inputs], rois_z)

        template = [bbox_feats_z[rois_z[:, 0] == j]
                    for j in range(len(gt_bboxes_z))]

        # Dual-Semantic Learning
        x_corr = next(self.rpn_similarity_learning(template, x))[0]

        # Concat x and x_corr
        x_concat = [x[i] + x_corr[i] for i in range(len(x))]
        # x_concat = tuple(x_concat)
        x = tuple(x_concat)

        # # Dual-Semantic RPN forward and loss
        # rpn_outs = self.rpn_head(x_concat)
        rpn_outs = self.rpn_head(x)

        # 设置gt_labels=None，相当于去除gt_labels
        rpn_loss_inputs = rpn_outs + (gt_bboxes_x, img_meta_x)

        rpn_losses = self.rpn_head.loss(*rpn_loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
        losses.update(rpn_losses)

        # parse proposal list
        proposal_inputs = rpn_outs + (img_meta_x, proposal_cfg)
        proposal_list = self.rpn_head.get_bboxes(*proposal_inputs)

        # assign gts and sample proposals
        if gt_bboxes_ignore is None:
            gt_bboxes_ignore = [None]
        assign_result = bbox_assigner.assign(
            proposal_list[0],
            gt_bboxes_x[0],
            gt_bboxes_ignore[0],
            gt_labels[0])
        sampling_result = bbox_sampler.sample(
            assign_result,
            proposal_list[0],
            gt_bboxes_x[0],
            gt_labels[0],
            feats=[lvl_feat[0][None] for lvl_feat in x])
        sampling_results = [sampling_result]

        # bbox head forward of RCNN
        rois_x = bbox2roi([res.bboxes for res in sampling_results])
        bbox_feats_x = self.roi_head.bbox_roi_extractor(
            x[:self.roi_head.bbox_roi_extractor.num_inputs], rois_x)

        # calculate bbox losses
        bbox_targets = self.roi_head.bbox_head.get_targets(
            sampling_results,
            gt_bboxes_x,
            gt_labels,
            self.train_cfg.rcnn)

        # Versatile learning
        bbox_feats_corr = self.rcnn_similarity_learning(
            bbox_feats_z, bbox_feats_x)
        # # Concat x and bbox_feats_corr
        # bbox_feats_fuse = bbox_feats_corr + bbox_feats_x
        # bbox_feats = bbox_feats_fuse

        if self.roi_head.with_shared_head:
            bbox_feats_corr = self.roi_head.shared_head(bbox_feats_corr)

        cls_score_corr, bbox_pred_corr = self.roi_head.bbox_head(bbox_feats_corr)

        loss_bbox_corr = self.roi_head.bbox_head.loss(
            cls_score_corr, bbox_pred_corr, rois_x, *bbox_targets)

        bbox_feats_x = bbox_feats_x
        if self.roi_head.with_shared_head:
            bbox_feats_x = self.roi_head.shared_head(bbox_feats_x)

        cls_score_x, bbox_pred_x = self.roi_head.bbox_head(bbox_feats_x)

        loss_bbox = self.roi_head.bbox_head.loss(
            cls_score_x, bbox_pred_x, rois_x, *bbox_targets)

        # update losses
        for k, v in loss_bbox_corr.items():
            if k in loss_bbox:
                if isinstance(v, (tuple, list)):
                    for u in range(len(v)):
                        loss_bbox[k][u] += v[u]
                else:
                    loss_bbox[k] += v
            else:
                loss_bbox[k] = v

        losses.update(loss_bbox)

        return losses

    def simple_test_bboxes(self,
                           bbox_feats_z,
                           x,
                           img_meta_x,
                           proposals,
                           rcnn_test_cfg,
                           rescale=False,
                           keep_order=False,
                           **kwargs):

        # bbox head forward of RCNN
        rois_x = bbox2roi(proposals)
        bbox_feats_x = self.roi_head.bbox_roi_extractor(
            x[:self.roi_head.bbox_roi_extractor.num_inputs], rois_x)

        # Versatile learning
        bbox_feats_corr = self.rcnn_similarity_learning(
            bbox_feats_z, bbox_feats_x)

        # # Concat x and bbox_feats_corr
        # bbox_feats_fuse = bbox_feats_corr + bbox_feats_x
        # bbox_feats_corr = bbox_feats_fuse

        if self.roi_head.with_shared_head:
            bbox_feats_corr = self.roi_head.shared_head(bbox_feats_corr)
        cls_score_corr, bbox_pred_corr = self.roi_head.bbox_head(bbox_feats_corr)

        # get predictions
        img_shape = img_meta_x[0]['img_shape']
        scale_factor = img_meta_x[0]['scale_factor']

        if keep_order:
            tra_bboxes, tra_labels = self.roi_head.bbox_head.get_bboxes(
                rois_x,
                cls_score_corr,
                bbox_pred_corr,
                img_shape,
                scale_factor,
                rescale=rescale,
                cfg=None)
            tra_bboxes = tra_bboxes[:, 4:]
            tra_labels = tra_labels[:, 1]
        else:
            tra_bboxes, tra_labels = self.roi_head.bbox_head.get_bboxes(
                rois_x,
                cls_score_corr,
                bbox_pred_corr,
                img_shape,
                scale_factor,
                rescale=rescale,
                cfg=rcnn_test_cfg)

        if self.roi_head.with_shared_head:
            bbox_feats_x = self.roi_head.shared_head(bbox_feats_x)
        cls_score_x, bbox_pred_x = self.roi_head.bbox_head(bbox_feats_x)
        if keep_order:
            det_bboxes, det_labels = self.roi_head.bbox_head.get_bboxes(
                rois_x,
                cls_score_x,
                bbox_pred_x,
                img_shape,
                scale_factor,
                rescale=rescale,
                cfg=None)
            det_bboxes = det_bboxes[:, 4:]
            det_labels = det_labels[:, 1]
        else:
            det_bboxes, det_labels = self.roi_head.bbox_head.get_bboxes(
                rois_x,
                cls_score_x,
                bbox_pred_x,
                img_shape,
                scale_factor,
                rescale=rescale,
                cfg=rcnn_test_cfg)

        return tra_bboxes, tra_labels, det_bboxes, det_labels

    def simple_matching(self,
                        bbox_feats_z,
                        bbox_feats_x,
                        rois_x,
                        img_meta_x,
                        rcnn_test_cfg,
                        rescale=False,
                        keep_order=False,
                        **kwargs):

        # similarity learning
        roi_feats = self.rcnn_similarity_learning(
            bbox_feats_z, bbox_feats_x)

        if self.roi_head.with_shared_head:
            roi_feats = self.roi_head.shared_head(roi_feats)
        cls_score, bbox_pred = self.roi_head.bbox_head(roi_feats)

        # get predictions
        img_shape = img_meta_x[0]['img_shape']
        scale_factor = img_meta_x[0]['scale_factor']

        if keep_order:
            match_bboxes, match_labels = self.roi_head.bbox_head.get_bboxes(
                rois_x,
                cls_score,
                bbox_pred,
                img_shape,
                scale_factor,
                rescale=rescale,
                cfg=None)
            match_bboxes = match_bboxes[:, 4:]
            match_labels = match_labels[:, 1]
        else:
            match_bboxes, match_labels = self.roi_head.bbox_head.get_bboxes(
                rois_x,
                cls_score,
                bbox_pred,
                img_shape,
                scale_factor,
                rescale=rescale,
                cfg=rcnn_test_cfg)

        return match_bboxes, match_labels

    def aug_test(self, *args, **kwargs):
        raise NotImplementedError(
            'aug_test is not implemented for SiamDTRCNN')

    def show_result(self, *args, **kwargs):
        raise NotImplementedError(
            'show_result is not implemented for SiamDTRCNN')

    def computeiou(self, bbox1, bbox2):
        """
        Calculates the intersection-over-union of two bounding boxes.
        Args:
            bbox1 (numpy.array, list of floats): bounding box in format x1,y1,x2,y2.
            bbox2 (numpy.array, list of floats): bounding box in format x1,y1,x2,y2.
        Returns:
            int: intersection-over-onion of bbox1, bbox2
        """
        bbox1 = [float(x) for x in bbox1]
        bbox2 = [float(x) for x in bbox2]

        (x0_1, y0_1, x1_1, y1_1) = bbox1
        (x0_2, y0_2, x1_2, y1_2) = bbox2

        # get the overlap rectangle
        overlap_x0 = max(x0_1, x0_2)
        overlap_y0 = max(y0_1, y0_2)
        overlap_x1 = min(x1_1, x1_2)
        overlap_y1 = min(y1_1, y1_2)

        # check if there is an overlap
        if overlap_x1 - overlap_x0 <= 0 or overlap_y1 - overlap_y0 <= 0:
            return 0

        # if yes, calculate the ratio of the overlap to each ROI size and the unified size
        size_1 = (x1_1 - x0_1) * (y1_1 - y0_1)
        size_2 = (x1_2 - x0_2) * (y1_2 - y0_2)
        size_intersection = (overlap_x1 - overlap_x0) * (overlap_y1 - overlap_y0)
        size_union = size_1 + size_2 - size_intersection

        return size_intersection / size_union

    def _process_query(self, img_z, gt_bboxes_z, img_meta_z):
        self._frame = 1
        self._learning_rate = 0.01
        self._img_meta_z = img_meta_z

        z = self.extract_feat(img_z)
        # get template
        rois_z = bbox2roi(gt_bboxes_z)
        self._bbox_feats_z = self.roi_head.bbox_roi_extractor(
            z[:self.roi_head.bbox_roi_extractor.num_inputs], rois_z)

        self._template = [self._bbox_feats_z[rois_z[:, 0] == j]
                          for j in range(len(gt_bboxes_z))]

        proposal_list = self.rpn_head.simple_test_rpn(
            z, img_meta_z)

        # proposal_list_sel=[proposal_list[0][:50,:]]
        proposal_list_sel = proposal_list

        # # delete the target region
        bgproposals = []
        for proposal in proposal_list_sel[0]:

            bbox = proposal[:-1]

            iouvalue = self.computeiou(bbox, gt_bboxes_z[0][0])

            if iouvalue <= 0:
                bgproposals.append(proposal.unsqueeze(0))

            if len(bgproposals) > 10:
                break

        # bgproposals=np.array(bgproposals)
        bgproposals = [jt.concat(bgproposals, dim=0)]

        # extract the features
        self._rois_bg = bbox2roi(bgproposals)
        self._bbox_feats_bg = self.roi_head.bbox_roi_extractor(
            z[:self.roi_head.bbox_roi_extractor.num_inputs], self._rois_bg)

    def _update_query(self, feat, proposal_list, gt_bboxes_z, img_meta_z):

        self._img_meta_z = img_meta_z

        # get template
        rois_z = bbox2roi(gt_bboxes_z)
        bbox_feats_z = self.roi_head.bbox_roi_extractor(
            feat[:self.roi_head.bbox_roi_extractor.num_inputs], rois_z)

        self._bbox_feats_z = (1 - self._learning_rate) * self._bbox_feats_z + self._learning_rate * bbox_feats_z

        self._template = [self._bbox_feats_z[rois_z[:, 0] == j]
                          for j in range(len(gt_bboxes_z))]

        # proposal_list_sel=[proposal_list[0][:50,:]]
        proposal_list_sel = proposal_list

        # # delete the target region
        bgproposals = []
        for proposal in proposal_list_sel[0]:

            bbox = proposal[:-1]
            iouvalue = self.computeiou(bbox, gt_bboxes_z[0][0])
            if iouvalue <= 0:
                bgproposals.append(proposal.unsqueeze(0))
            if len(bgproposals) > 10:
                break

        bgproposals = [jt.concat(bgproposals, dim=0)]

        # extract the features
        self._rois_bg = bbox2roi(bgproposals)
        self._bbox_feats_bg = self.roi_head.bbox_roi_extractor(
            feat[:self.roi_head.bbox_roi_extractor.num_inputs], self._rois_bg)

    def _process_gallary(self, img_x, img_meta_x, **kwargs):

        self._frame = self._frame + 1

        x_src = self.extract_feat(img_x)

        x = copy.deepcopy(x_src)

        # Dual-Semantic Learning
        x_corr = next(self.rpn_similarity_learning(self._template, x))[0]

        # Concat x and x_corr
        x_concat = [x[i] + x_corr[i] for i in range(len(x))]
        x = tuple(x_concat)

        proposal_list = self.rpn_head.simple_test_rpn(
            x, img_meta_x)

        # proposal_list=[proposal_list[0][:50,:]]

        # RCNN forward
        tra_bboxes, tra_labels, det_bboxes, det_labels = self.simple_test_bboxes(
            self._bbox_feats_z, x, img_meta_x, proposal_list, self.test_cfg.rcnn, **kwargs)

        ens_bboxes = jt.concatcat((tra_bboxes, det_bboxes), dim=0)
        ens_labels = jt.concat((tra_labels, det_labels), dim=0)

        # choose top results for background suppression
        Top_NUM = 5
        ens_scores = ens_bboxes[:, -1]
        ens_values, ens_indices = jt.sort(ens_scores, descending=True)
        newens_bboxes = ens_bboxes[ens_indices][:Top_NUM]
        newens_labels = ens_labels[ens_indices][:Top_NUM]

        # background suppression
        for mm in range(0, Top_NUM):
            mm_bbox = newens_bboxes[mm:mm + 1, :-1]

            # bbox head forward of mm_query
            rois_mm = bbox2roi([mm_bbox])
            bbox_feats_mm = self.roi_head.bbox_roi_extractor(
                x[:self.roi_head.bbox_roi_extractor.num_inputs], rois_mm)

            # matching RCNN
            mm_bboxes, mm_labels = self.simple_matching(
                bbox_feats_mm, self._bbox_feats_bg, self._rois_bg,
                self._img_meta_z, self.test_cfg.rcnn, **kwargs)

            mm_scores = mm_bboxes[:, -1]
            mm_max_scores = max(mm_scores)
            newens_bboxes[mm, -1] = newens_bboxes[mm, -1] - mm_max_scores

        ref_scores = newens_bboxes[:, -1].clone()

        for ii in range(0, Top_NUM):

            bboxA = newens_bboxes[ii:ii + 1, :-1]
            # Enhanced with IOU
            for jj in range(0, Top_NUM):

                if ii == jj:
                    continue
                else:

                    bboxB = newens_bboxes[jj:jj + 1, :-1]
                    # import pdb;pdb.set_trace()
                    iouvalue = self.computeiou(bboxA[0], bboxB[0])

                    if iouvalue > 0.8:
                        newens_bboxes[ii, -1] = newens_bboxes[ii, -1] + ref_scores[jj] * iouvalue

        # update flag
        if tra_bboxes[0,-1]+det_bboxes[0,-1]>1.9 and self.computeiou(det_bboxes[0, :-1], tra_bboxes[0, :-1])>0.8:

            up_flag = True
            new_bbox = det_bboxes[0:0+1, :-1]*img_meta_x[0]['scale_factor']
            self._update_query(x_src, proposal_list, [new_bbox], img_meta_x)

        else:
            up_flag = False
            pass

        if not kwargs.get('keep_order', False):
            bbox_results = bbox2result(
                newens_bboxes, newens_labels, self.roi_head.bbox_head.num_classes)
        else:
            bbox_results = [np.concatenate([
                newens_bboxes.cpu().numpy(),
                newens_labels.cpu().numpy()[:, None]], axis=1)]

        return bbox_results[0], up_flag
