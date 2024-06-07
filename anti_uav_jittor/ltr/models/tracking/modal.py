import copy
import math

# import torch.nn as nn
from jittor import nn
from util.misc import accuracy
from util import box_ops
from ltr.models.loss.matcher import build_matcher
from ltr.models.neck.my_position_encoding import build_position_encoding_modal
from ltr.models.neck.Modal_fusion import build_featurefusion_network
from ltr import model_constructor
from ltr.models.neck.Modal_fusion import ModalFusionMultiScale,FeatureFusionNetwork_FPN
from ltr.models.neck.prediction_neck import PredictionNeck
import jittor as jt
class Modal_FPN(nn.Module):
    def __init__(self, backbone, posiont_encoding):
        super(Modal_FPN, self).__init__()
        self.backbone_ir = copy.deepcopy(backbone)
        self.backbone_rgb = copy.deepcopy(backbone)
        posiont_encoding_single = posiont_encoding['single']
        posiont_encoding_multiple = posiont_encoding['multiple']
        self.fusion_block = ModalFusionMultiScale(position_encodeing=posiont_encoding_single)
        self.correlation = FeatureFusionNetwork_FPN(position_encoding=posiont_encoding_multiple)
        hidden_dim = 512
        num_classes = 1
        self.prediction_head = PredictionNeck(hidden_dim=hidden_dim,num_layers=3)
    def execute(self, search, template):
        search_ir,search_rgb = search['ir'],search['rgb']
        template_ir, template_rgb = template['ir'], template['rgb']#4.256.16.16
        feature_template_rgb = self.backbone_rgb(template_rgb)#torch.size([b,256,h,w]) *3
        feature_search_rgb = self.backbone_rgb(search_rgb)
        feature_template_ir = self.backbone_ir(template_ir)
        feature_search_ir = self.backbone_ir(search_ir)

        fusion_feature_search = self.fusion_block(feature_search_rgb,feature_search_ir)
        fusion_feature_tmplate = self.fusion_block(feature_template_rgb,feature_template_ir)

        hs = self.correlation(fusion_feature_search,fusion_feature_tmplate)
        out_put = self.prediction_head(hs)
        return out_put

    def template(self,template):
        template_ir, template_rgb = template['ir'], template['rgb']
        feature_template_rgb = self.backbone_rgb(template_rgb)#torch.size([b,256,h,w]) *3
        feature_template_ir = self.backbone_ir(template_ir)
        zf_rbg = feature_template_rgb
        zf_ir = feature_template_ir
        self.template_fusion = self.fusion_block(zf_rbg,zf_ir)

    def track(self,search_ir,search_rgb):

        feature_search_rgb = self.backbone_rgb(search_rgb)#torch.size([b,256,h,w]) *3
        feature_search_ir = self.backbone_ir(search_ir)

        fusion_feature_search = self.fusion_block(feature_search_rgb,feature_search_ir)
        template_fusion = self.template_fusion
        hs = self.correlation(fusion_feature_search, template_fusion)
        out_put = self.prediction_head(hs)
        return out_put








from ltr.models.backbone.Modal_backbone import build_backbone
from ltr.models.neck.my_position_encoding import build_position_encoding_modal


import ltr.admin.settings as ws_settings


class SetCriterion(nn.Module):
    """ This class computes the loss for TransT.
    The process happens in two steps:
        1) we compute assignment between ground truth box and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """
    def __init__(self, num_classes, matcher, weight_dict, eos_coef, losses):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, always be 1 for single object tracking.
            matcher: module able to compute a matching between target and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        empty_weight = jt.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer('empty_weight', empty_weight)

    def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = jt.concat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = jt.full(src_logits.shape[:2], self.num_classes,
                                    dtype=jt.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
        losses = {'loss_ce': loss_ce}

        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, h, w), normalized by the image size.
        """
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = jt.concat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        loss_bbox = nn.l1_loss(src_boxes, target_boxes, reduction='none')

        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes
        giou, iou = box_ops.generalized_box_iou(
            box_ops.box_cxcywh_to_xyxy(src_boxes),
            box_ops.box_cxcywh_to_xyxy(target_boxes))
        giou = jt.diag(giou)
        iou = jt.diag(iou)
        loss_giou = 1 - giou
        iou = iou
        losses['loss_giou'] = loss_giou.sum() / num_boxes
        losses['iou'] = iou.sum() / num_boxes
        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = jt.cat([jt.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = jt.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = jt.cat([jt.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = jt.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes):
        loss_map = {
            'labels': self.loss_labels,
            'boxes': self.loss_boxes
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes)

    def execute(self, outputs, targets):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}

        # Retrieve the matching between the outputs of the last layer and the target
        indices = self.matcher(outputs_without_aux, targets)

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes_pos = sum(len(t[0]) for t in indices)

        num_boxes_pos = jt.Var([num_boxes_pos], dtype=jt.float, device=next(iter(outputs.values())).device)

        num_boxes_pos = jt.clamp(num_boxes_pos, min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes_pos))

        return losses


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def execute(self, x):
        for i, layer in enumerate(self.layers):
            x = nn.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x





@model_constructor
def modal_swin_fpn(settings):
    backbone_name = settings.backbone_name
    backbone = build_backbone(swin_name=backbone_name)
    postion_encoding = build_position_encoding_modal(settings)
    model = Modal_FPN(backbone=backbone,posiont_encoding=postion_encoding)
    model
    return model

def Modal_loss(settings):
    num_classes = 1
    matcher = build_matcher()
    weight_dict = {'loss_ce': 6, 'loss_bbox': 4}
    weight_dict['loss_giou'] = 7
    losses = ['labels', 'boxes']
    criterion = SetCriterion(num_classes, matcher=matcher, weight_dict=weight_dict,
                             eos_coef=0.0625, losses=losses)
    criterion
    return criterion


settings = ws_settings.Settings()
settings.position_embedding = 'sine'
settings.device = 'cuda'
settings.backbone_name = 'swin_base_patch4_window12_384'

settings.hidden_dim = 256

# net = modal_swin_fpn(settings)
# net.cuda()
# x1 = torch.rand((2,3,256,256)).cuda()
# x2 = torch.rand((2,3,256,256)).cuda()
# x3 = torch.rand((2,3,128,128)).cuda()
# x4 = torch.rand((2,3,128,128)).cuda()
#
#
#
# search  ={'rgb':x1,'ir':x2}
# template = {'rgb':x3,'ir':x4}
# y = net(search,template)
# for layer in y:
#     print(layer)
# print('g')




'''
torch.Size([2, 64, 512])
torch.Size([2, 256, 512])
torch.Size([2, 1024, 512])
'''