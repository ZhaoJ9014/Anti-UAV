
from ltr import model_constructor
import copy

from util import box_ops
from util.misc import (NestedTensor, nested_tensor_from_tensor,
                       nested_tensor_from_tensor_2,
                       accuracy)

from ltr.models.backbone.transt_backbone import build_backbone
from ltr.models.loss.matcher import build_matcher
from ltr.models.neck.featurefusion_network import build_featurefusion_network
from ltr.models.neck.UnFPNmodality_fusion import ModalFusionNetwork
from jittor import nn
import jittor as jt

class UnFPN(nn.Module):
    """ This is the TransT module that performs single object tracking """
    def __init__(self, backbone, featurefusion_network, num_classes):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See transt_backbone.py
            featurefusion_network: torch module of the featurefusion_network architecture, a variant of transformer.
                                   See featurefusion_network.py
            num_classes: number of object classes, always 1 for single object tracking
        """
        super().__init__()
        self.featurefusion_network = featurefusion_network
        hidden_dim = featurefusion_network.d_model
        self.class_embed = MLP(hidden_dim, hidden_dim, num_classes + 1, 3)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.input_proj = nn.Conv2d(backbone.num_channels, 256, kernel_size=1)
        self.backbone_ir = copy.deepcopy(backbone)
        self.backbone_rgb = copy.deepcopy(backbone)
        self.modality_fusion_search = ModalFusionNetwork()
        self.modality_fusion_template = ModalFusionNetwork()
        self.prediction_head = PredictionNeck(hidden_dim=hidden_dim, num_layers=3)

    def forward(self, search, template):
        """ The forward expects a NestedTensor, which consists of:
               - search.tensors: batched images, of shape [batch_size x 3 x H_search x W_search]
               - search.mask: a binary mask of shape [batch_size x H_search x W_search], containing 1 on padded pixels
               - template.tensors: batched images, of shape [batch_size x 3 x H_template x W_template]
               - template.mask: a binary mask of shape [batch_size x H_template x W_template], containing 1 on padded pixels

            It returns a dict with the following elements:
               - "pred_logits": the classification logits for all feature vectors.
                                Shape= [batch_size x num_vectors x (num_classes + 1)]
               - "pred_boxes": The normalized boxes coordinates for all feature vectors, represented as
                               (center_x, center_y, height, width). These values are normalized in [0, 1],
                               relative to the size of each individual image.

        """
        search_ir, search_rgb = search['ir'], search['rgb'] #search : b*3*256*256
        template_ir, template_rgb = template['ir'], template['rgb']

        search_ir = nested_tensor_from_tensor(search_ir)
        search_rgb =  nested_tensor_from_tensor(search_rgb)
        template_ir = nested_tensor_from_tensor(template_ir)
        template_rgb = nested_tensor_from_tensor(template_rgb)


        feature_template_rgb, pos_template_rgb = self.backbone_rgb(template_rgb) # b*1024*16*16,positon b*256*16*16
        feature_search_rgb,pos_search_rgb = self.backbone_rgb(search_rgb)# b*1024*32*32,positon b*256*32*32
        feature_template_ir, pos_template_ir = self.backbone_ir(template_ir)
        feature_search_ir,pos_search_ir = self.backbone_ir(search_ir)
        # modality_fusion
        search = self.modality_fusion_search(self.input_proj(feature_search_rgb[-1].tensors), self.input_proj(feature_search_ir[-1].tensors),
                                             pos_search_rgb[0], pos_search_ir[0],flatten=True)
        template = self.modality_fusion_template(self.input_proj(feature_template_rgb[-1].tensors), self.input_proj(feature_template_ir[-1].tensors),
                                             pos_template_rgb[0], pos_template_ir[0],flatten=True)
        hs = self.featurefusion_network(search, None, template, None, None, None)

        out_put = self.prediction_head(hs)
        return out_put

    def template(self,template):
        template_ir, template_rgb = template['ir'], template['rgb']
        template_ir = nested_tensor_from_tensor(template_ir)
        template_rgb = nested_tensor_from_tensor(template_rgb)
        feature_template_rgb, pos_template_rgb = self.backbone_rgb(template_rgb)#torch.size([b,256,h,w]) *3
        feature_template_ir, pos_template_ir = self.backbone_ir(template_ir)


        self.template_fusion = self.modality_fusion_template(self.input_proj(feature_template_rgb[-1].tensors), self.input_proj(feature_template_ir[-1].tensors),
                                             pos_template_rgb[0], pos_template_ir[0],flatten=True)

    def track(self,search_ir,search_rgb):
        search_ir = nested_tensor_from_tensor(search_ir)
        search_rgb = nested_tensor_from_tensor(search_rgb)
        feature_search_rgb,pos_search_rgb = self.backbone_rgb(search_rgb)
        feature_search_ir,pos_search_ir = self.backbone_ir(search_ir)

        fusion_feature_search = self.modality_fusion_search(self.input_proj(feature_search_rgb[-1].tensors),
                                             self.input_proj(feature_search_ir[-1].tensors),
                                             pos_search_rgb[0], pos_search_ir[0], flatten=True)
        template_fusion = self.template_fusion
        hs = self.featurefusion_network(fusion_feature_search, None, template_fusion, None, None, None)
        out_put = self.prediction_head(hs)
        return out_put


class PredictionNeck(nn.Module):
    def __init__(self,hidden_dim,num_layers):
        super(PredictionNeck, self).__init__()
        hidden_dim = 512
        num_classes = 1
        self.class_embed_rgb = MLP(hidden_dim, hidden_dim, num_classes + 1, 3)
        self.bbox_embed_rgb = MLP(hidden_dim, hidden_dim, 4, 3)
        self.class_embed_ir = MLP(hidden_dim, hidden_dim, num_classes + 1, 3)
        self.bbox_embed_ir = MLP(hidden_dim, hidden_dim, 4, 3)

    def forward(self,hs):
        outputs_class_rgb = self.class_embed_rgb(hs)  #1,2,512,2
        outputs_class_ir = self.class_embed_ir(hs)
        outputs_coord_rgb = self.bbox_embed_rgb(hs).sigmoid()
        outputs_coord_ir = self.bbox_embed_ir(hs).sigmoid()
        out = []
        out.append({'pred_logits': outputs_class_rgb, 'pred_boxes': outputs_coord_rgb})
        out.append({'pred_logits': outputs_class_ir, 'pred_boxes': outputs_coord_ir})
        return out





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
        batch_idx = jt.concat([jt.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = jt.concat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = jt.concat([jt.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = jt.concat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes):
        loss_map = {
            'labels': self.loss_labels,
            'boxes': self.loss_boxes
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes)

    def forward(self, outputs, targets):
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

        num_boxes_pos = jt.var([num_boxes_pos], dtype=jt.float, device=next(iter(outputs.values())).device)

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

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = nn.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


@model_constructor
def unFPN_resnet50(settings):
    num_classes = 1
    backbone_net = build_backbone(settings, backbone_pretrained=True)
    featurefusion_network = build_featurefusion_network(settings)
    model = UnFPN(
        backbone_net,
        featurefusion_network,
        num_classes=num_classes
    )
    model
    return model

def transt_loss(settings):
    num_classes = 1
    matcher = build_matcher()
    weight_dict = {'loss_ce': 8.334, 'loss_bbox': 5}
    weight_dict['loss_giou'] = 2
    losses = ['labels', 'boxes']
    criterion = SetCriterion(num_classes, matcher=matcher, weight_dict=weight_dict,
                             eos_coef=0.0625, losses=losses)
    criterion
    return criterion


