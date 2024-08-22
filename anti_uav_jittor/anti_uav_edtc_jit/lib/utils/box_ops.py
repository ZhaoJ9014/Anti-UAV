import math
import jittor as jt
# from jtvision.ops.boxes import box_area
import numpy as np
# from jt.nn import functional as F
from jittor import nn


def box_area(boxes):
    """
    Compute the area of a set of bounding boxes.

    Args:
        boxes (jt.Var): A tensor of shape (N, 4) where N is the number of boxes and the
            boxes are specified in the format [x_min, y_min, x_max, y_max].

    Returns:
        area (jt.Var): A tensor of shape (N,) representing the area for each box.
    """
    # 计算每个边界框的宽度和高度
    width = boxes[:, 2] - boxes[:, 0]
    height = boxes[:, 3] - boxes[:, 1]
    
    # 计算面积
    area = width * height
    
    return area

def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return jt.stack(b, dim=-1)


def box_xywh_to_xyxy(x):
    x1, y1, w, h = x.unbind(-1)
    b = [x1, y1, x1 + w, y1 + h]
    return jt.stack(b, dim=-1)


def box_xyxy_to_xywh(x):
    x1, y1, x2, y2 = x.unbind(-1)
    b = [x1, y1, x2 - x1, y2 - y1]
    return jt.stack(b, dim=-1)


def box_xyxy_to_cxcywh(x):
    x0, y0, x1, y1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2,
         (x1 - x0), (y1 - y0)]
    return jt.stack(b, dim=-1)


# modified from jtvision to also return the union
'''Note that this function only supports shape (N,4)'''


def box_iou(boxes1, boxes2):
    """

    :param boxes1: (N, 4) (x1,y1,x2,y2)
    :param boxes2: (N, 4) (x1,y1,x2,y2)
    :return:
    """
    area1 = box_area(boxes1) # (N,)
    area2 = box_area(boxes2) # (N,)

    lt = jt.max(boxes1[:, :2], boxes2[:, :2])  # (N,2)
    rb = jt.min(boxes1[:, 2:], boxes2[:, 2:])  # (N,2)

    wh = (rb - lt).clamp(min=0)  # (N,2)
    inter = wh[:, 0] * wh[:, 1]  # (N,)

    union = area1 + area2 - inter

    iou = inter / union
    return iou, union


'''Note that this implementation is different from DETR's'''


def generalized_box_iou(boxes1, boxes2):
    """
    Generalized IoU from https://giou.stanford.edu/

    The boxes should be in [x0, y0, x1, y1] format

    boxes1: (N, 4)
    boxes2: (N, 4)
    """
    # degenerate boxes gives inf / nan results
    # so do an early check
    # try:
    assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
    assert (boxes2[:, 2:] >= boxes2[:, :2]).all()
    iou, union = box_iou(boxes1, boxes2) # (N,)

    lt = jt.min(boxes1[:, :2], boxes2[:, :2])
    rb = jt.max(boxes1[:, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(min=0)  # (N,2)
    area = wh[:, 0] * wh[:, 1] # (N,)

    return iou - (area - union) / area, iou


def giou_loss(boxes1, boxes2):
    """

    :param boxes1: (N, 4) (x1,y1,x2,y2)
    :param boxes2: (N, 4) (x1,y1,x2,y2)
    :return:
    """
    giou, iou = generalized_box_iou(boxes1, boxes2)
    return (1 - giou).mean(), iou



def ciou_loss(bboxes1, bboxes2):
    """
    :param boxes1: (N, 4) (x1,y1,x2,y2)
    :param boxes2: (N, 4) (x1,y1,x2,y2)
    :return:
    """
    rows = bboxes1.shape[0]
    cols = bboxes2.shape[0]
    cious = jt.zeros((rows, cols))
    if rows * cols == 0:
        return cious
    exchange = False
    if bboxes1.shape[0] > bboxes2.shape[0]:
        bboxes1, bboxes2 = bboxes2, bboxes1
        cious = jt.zeros((cols, rows))
        exchange = True
    w1 = bboxes1[:, 2] - bboxes1[:, 0]
    h1 = bboxes1[:, 3] - bboxes1[:, 1]
    w2 = bboxes2[:, 2] - bboxes2[:, 0]
    h2 = bboxes2[:, 3] - bboxes2[:, 1]
    area1 = w1 * h1
    area2 = w2 * h2
    center_x1 = (bboxes1[:, 0] + bboxes1[:, 2]) / 2.
    center_y1 = (bboxes1[:, 1] + bboxes1[:, 3]) / 2.
    center_x2 = (bboxes2[:, 0] + bboxes2[:, 2]) / 2.
    center_y2 = (bboxes2[:, 1] + bboxes2[:, 3]) / 2.

    inter_l = jt.max(center_x1 - w1 / 2,center_x2 - w2 / 2)
    inter_r = jt.min(center_x1 + w1 / 2,center_x2 + w2 / 2)
    inter_t = jt.max(center_y1 - h1 / 2,center_y2 - h2 / 2)
    inter_b = jt.min(center_y1 + h1 / 2,center_y2 + h2 / 2)
    inter_area = jt.clamp((inter_r - inter_l),min=0) * jt.clamp((inter_b - inter_t),min=0)

    c_l = jt.min(center_x1 - w1 / 2,center_x2 - w2 / 2)
    c_r = jt.max(center_x1 + w1 / 2,center_x2 + w2 / 2)
    c_t = jt.min(center_y1 - h1 / 2,center_y2 - h2 / 2)
    c_b = jt.max(center_y1 + h1 / 2,center_y2 + h2 / 2)

    inter_diag = (center_x2 - center_x1)**2 + (center_y2 - center_y1)**2
    c_diag = jt.clamp((c_r - c_l),min=0)**2 + jt.clamp((c_b - c_t),min=0)**2

    union = area1+area2-inter_area
    u = (inter_diag) / c_diag
    iou = inter_area / union
    v = (4 / (math.pi ** 2)) * jt.pow((jt.atan(w2 / h2) - jt.atan(w1 / h1)), 2)
    with jt.no_grad():
        S = (iou>0.5).float()
        alpha= S*v/(1-iou+v)
    cious = iou - u - alpha * v
    cious = jt.clamp(cious,min=-1.0,max = 1.0)
    if exchange:
        cious = cious.T
    return jt.mean(1-cious), iou


def clip_box(box: list, H, W, margin=0):
    x1, y1, w, h = box
    x2, y2 = x1 + w, y1 + h
    x1 = min(max(0, x1), W-margin)
    x2 = min(max(margin, x2), W)
    y1 = min(max(0, y1), H-margin)
    y2 = min(max(margin, y2), H)
    w = max(margin, x2-x1)
    h = max(margin, y2-y1)
    return [x1, y1, w, h]


class REGLoss(nn.Module):
    def __init__(self, dim=4, loss_type='iou'):
        super(REGLoss, self).__init__()
        self.dim = dim
        if loss_type == 'iou':
            self.loss = IOULoss()
        else:
            raise ValueError("Only support iou loss.")

    def forward(self, output, ind, target, radius=1, norm=1/20.):
        width, height = output.size(-2), output.size(-1)
        output = output.view(-1, self.dim, width, height)
        # mask =  mask.view(-1, 2)
        target = target.view(-1, self.dim)
        ind = ind.view(-1, 1)
        center_w = (ind % width).int().float()
        center_h = (ind / width).int().float()

        # regress for the coordinates in the vicinity of the target center, the default radius is 1.
        if radius is not None:
            loss = []
            for r_w in range(-1 * radius, radius + 1):
                for r_h in range(-1 * radius, radius + 1):
                    target_wl = target[:, 0] + r_w * norm
                    target_wr = target[:, 1] - r_w * norm
                    target_ht = target[:, 2] + r_h * norm
                    target_hb = target[:, 3] - r_h * norm
                    if (target_wl < 0.).any() or (target_wr < 0.).any() or (target_ht < 0.).any() or (target_hb < 0.).any():
                        continue
                    if (center_h + r_h < 0.).any() or (center_h + r_h >= 1.0 * width).any() \
                            or (center_w + r_w < 0.).any() or (center_w + r_w >= 1.0 * width).any():
                        continue

                    target_curr = jt.stack((target_wl, target_wr, target_ht, target_hb), dim=1)  # [num_images * num_sequences, 4]
                    ind_curr = ((center_h + r_h) * width + (center_w + r_w)).long()
                    pred_curr = _tranpose_and_gather_feat(output, ind_curr)
                    loss_curr = self.loss(pred_curr, target_curr)
                    loss.append(loss_curr)
            if len(loss) == 0:
                pred = _tranpose_and_gather_feat(output, ind.long())  # pred shape: [num_images * num_sequences, 4]
                loss = self.loss(pred, target)
                return loss
            loss = jt.stack(loss, dim=0)
            loss = jt.mean(loss, dim=0)
            return loss
        pred = _tranpose_and_gather_feat(output, ind.long())     # pred shape: [num_images * num_sequences, 4]
        loss = self.loss(pred, target)

        return loss

class IOULoss(nn.Module):
    def __init__(self, reduction='mean'):
        super(IOULoss, self).__init__()
        self.reduction = reduction

    def forward(self, pred, target, weight=None):
        pred_left = pred[:, 0]
        pred_top = pred[:, 2]
        pred_right = pred[:, 1]
        pred_bottom = pred[:, 3]

        target_left = target[:, 0]
        target_top = target[:, 2]
        target_right = target[:, 1]
        target_bottom = target[:, 3]

        target_area = (target_left + target_right) * \
                      (target_top + target_bottom)
        pred_area = (pred_left + pred_right) * \
                    (pred_top + pred_bottom)

        w_intersect = jt.min(pred_left, target_left) + \
                      jt.min(pred_right, target_right)
        h_intersect = jt.min(pred_bottom, target_bottom) + \
                      jt.min(pred_top, target_top)

        area_intersect = w_intersect * h_intersect
        area_union = target_area + pred_area - area_intersect

        losses = -jt.log((area_intersect + 1.0) / (area_union + 1.0))

        if weight is not None and weight.sum() > 0:
            return (losses * weight).sum() / weight.sum()
        else:
            assert losses.numel() != 0
            if self.reduction == 'mean':
                return losses.mean()
            else:
                return losses.sum()

def _tranpose_and_gather_feat(feat, ind):
    feat = feat.permute(0, 2, 3, 1).contiguous()
    feat = feat.view(feat.size(0), -1, feat.size(3))
    feat = _gather_feat(feat, ind)
    return feat

def _gather_feat(feat, ind, mask=None):
    dim = feat.size(2)
    ind = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
    feat = feat.gather(dim=1, index=ind)   # [num_images * num_sequences, 1, 2]
    if mask is not None:
        mask = mask.unsqueeze(2).expand_as(feat)
        feat = feat[mask]
        feat = feat.view(-1, dim)
    return feat.view(ind.size(0), dim)


class LBHinge(nn.Module):
    """Loss that uses a 'hinge' on the lower bound.
    This means that for samples with a label value smaller than the threshold, the loss is zero if the prediction is
    also smaller than that threshold.
    args:
        error_matric:  What base loss to use (MSE by default).
        threshold:  Threshold to use for the hinge.
        clip:  Clip the loss if it is above this value.
    """
    def __init__(self, error_metric=nn.MSELoss(), threshold=None, clip=None):
        super().__init__()
        self.error_metric = error_metric
        self.threshold = threshold if threshold is not None else -100
        self.clip = clip

    def forward(self, prediction, label, target_bb=None):
        # print("pred shape: {}, label shape: {}".format(prediction.shape, label.shape))
        negative_mask = (label < self.threshold).float()
        positive_mask = (1.0 - negative_mask)

        prediction = negative_mask * nn.relu(prediction) + positive_mask * prediction

        loss = self.error_metric(prediction, positive_mask * label)

        if self.clip is not None:
            loss = jt.min(loss, jt.var([self.clip], device=loss.device))
        return loss

# evidential Losses

def get_device():
    use_cuda = jt.cuda.is_available()
    device = jt.device("cuda:0" if use_cuda else "cpu")
    return device

def relu_evidence(y):
    return nn.relu(y)


def exp_evidence(y):
    return jt.exp(jt.clamp(y, -10, 10))


def softplus_evidence(y):
    return nn.softplus(y)

def kl_divergence(alpha, num_classes, device=None):
    if not device:
        device = get_device()
    ones = jt.ones([1, num_classes], dtype=jt.float32, device=device)
    sum_alpha = jt.sum(alpha, dim=1, keepdim=True)
    first_term = (
        jt.lgamma(sum_alpha)
        - jt.lgamma(alpha).sum(dim=1, keepdim=True)
        + jt.lgamma(ones).sum(dim=1, keepdim=True)
        - jt.lgamma(ones.sum(dim=1, keepdim=True))
    )
    second_term = (
        (alpha - ones)
        .mul(jt.digamma(alpha) - jt.digamma(sum_alpha))
        .sum(dim=1, keepdim=True)
    )
    kl = first_term + second_term
    return kl


def loglikelihood_loss(y, alpha, device=None):
    if not device:
        device = get_device()
    y = y.to(device)
    alpha = alpha.to(device)
    S = jt.sum(alpha, dim=1, keepdim=True)
    loglikelihood_err = jt.sum((y - (alpha / S)) ** 2, dim=1, keepdim=True)
    loglikelihood_var = jt.sum(
        alpha * (S - alpha) / (S * S * (S + 1)), dim=1, keepdim=True
    )
    loglikelihood = loglikelihood_err + loglikelihood_var
    return loglikelihood


def mse_loss(y, alpha, epoch_num, num_classes, annealing_step, device=None):
    if not device:
        device = get_device()
    y = y.to(device)
    alpha = alpha.to(device)
    loglikelihood = loglikelihood_loss(y, alpha, device=device)

    annealing_coef = jt.min(
        jt.var(1.0, dtype=jt.float32),
        jt.var(epoch_num / annealing_step, dtype=jt.float32),
    )

    kl_alpha = (alpha - 1) * (1 - y) + 1
    kl_div = annealing_coef * kl_divergence(kl_alpha, num_classes, device=device)
    return loglikelihood + kl_div


def edl_loss(func, y, alpha, epoch_num, num_classes, annealing_step, device=None):
    y = y.to(device)
    alpha = alpha.to(device)
    S = jt.sum(alpha, dim=1, keepdim=True)

    A = jt.sum(y * (func(S) - func(alpha)), dim=1, keepdim=True)

    annealing_coef = jt.min(
        jt.var(1.0, dtype=jt.float32),
        jt.var(epoch_num / annealing_step, dtype=jt.float32),
    )

    kl_alpha = (alpha - 1) * (1 - y) + 1
    kl_div = annealing_coef * kl_divergence(kl_alpha, num_classes, device=device)
    return A + kl_div


def edl_mse_loss(output, target, epoch_num, num_classes, annealing_step, device=None):
    if not device:
        device = get_device()
    if output.ndim==3:
        output = output.squeeze(1)
    evidence = relu_evidence(output)
    alpha = evidence + 1
    loss = jt.mean(
        mse_loss(target, alpha, epoch_num, num_classes, annealing_step, device=device)
    )
    return loss


def edl_log_loss(output, target, epoch_num, num_classes, annealing_step, device=None):
    if not device:
        device = get_device()
    evidence = relu_evidence(output)
    alpha = evidence + 1
    loss = jt.mean(
        edl_loss(
            jt.log, target, alpha, epoch_num, num_classes, annealing_step, device
        )
    )
    return loss


def edl_digamma_loss(
    output, target, epoch_num, num_classes, annealing_step, device=None
):
    if not device:
        device = get_device()
    evidence = relu_evidence(output)
    alpha = evidence + 1
    loss = jt.mean(
        edl_loss(
            jt.digamma, target, alpha, epoch_num, num_classes, annealing_step, device
        )
    )
    return loss