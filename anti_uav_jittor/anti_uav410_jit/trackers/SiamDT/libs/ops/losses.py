# import torch
# import torch.nn as nn
# import torch.nn.functional as F
import jittor as jt
import jittor.nn as nn

def binary_cross_entropy_with_logits(input, target, weight=None, reduction='mean'):
    """
    Args:
        input: Logits tensor, shape (N, *)
        target: Targets tensor, shape (N, *), same shape as input
        weight: Weight tensor, shape (N, *), same shape as input, optional
        reduction: 'none', 'mean', or 'sum'
    
    Returns:
        Loss tensor
    """
    # 使用 sigmoid 激活函数将 input 转换为概率
    sigmoid_input = jt.sigmoid(input)
    
    # 计算二元交叉熵损失
    loss = - (target * jt.log(sigmoid_input) + (1 - target) * jt.log(1 - sigmoid_input))
    
    # 如果提供了权重，应用权重
    if weight is not None:
        loss *= weight
    
    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    else:
        return loss

def _log_sigmoid(x):
    r'''Numerical stable log-sigmoid.
    '''
    return jt.clamp(x, max=0) - \
        jt.log(1 + jt.exp(-jt.abs(x))) + \
        0.5 * jt.clamp(x, min=0, max=0)


def _log_one_minus_sigmoid(x):
    r'''Numerical stable log-one-minus-sigmoid.
    '''
    return jt.clamp(-x, max=0) - \
        jt.log(1 + jt.exp(-jt.abs(x))) + \
        0.5 * jt.clamp(x, min=0, max=0)


def balanced_bce_loss(input, target, neg_weight=1.):
    target = target.type_as(input)
    pos_mask, neg_mask = target.eq(1), target.eq(0)
    pos_num = pos_mask.sum().float().clamp(1e-12)
    neg_num = neg_mask.sum().float().clamp(1e-12)
    weight = target.new_zeros(target.size())
    weight[pos_mask] = 1. / pos_num
    weight[neg_mask] = (1. / neg_num) * neg_weight
    weight /= weight.mean()
    return binary_cross_entropy_with_logits(
        input, target, weight, reduction='mean')


def focal_loss(input, target, alpha=0.25, gamma=2.):
    prob = input.sigmoid()
    target = target.type_as(input)
    
    # focal weights
    pt = (1 - prob) * target + prob * (1 - target)
    weight = pt.pow(gamma) * \
        (alpha * target + (1 - alpha) * (1 - target))
    
    # BCE loss with focal weights
    loss = binary_cross_entropy_with_logits(
        input, target, reduction='none') * weight

    return loss.mean()


def ghmc_loss(input, target, bins=30, momentum=0.5):
    edges = [t / bins for t in range(bins + 1)]
    edges[-1] += 1e-6
    mmt = momentum
    if mmt > 0:
        acc_sum = [0.0 for _ in range(bins)]
    
    weight = jt.zeros_like(input)
    # gradient length
    g = jt.abs(input.sigmoid().detach() - target)

    total = input.numel()
    n = 0
    for i in range(bins):
        inds = (g >= edges[i]) & (g < edges[i + 1])
        num_in_bin = inds.sum().item()
        if num_in_bin > 0:
            if mmt > 0:
                acc_sum[i] = mmt * acc_sum[i] \
                    + (1 - mmt) * num_in_bin
                weight[inds] = total / acc_sum[i]
            else:
                weight[inds] = total / num_in_bin
            n += 1
    if n > 0:
        weight /= weight.mean()

    loss = binary_cross_entropy_with_logits(
        input, target, weight, reduction='sum') / total
    
    return loss


def ohem_bce_loss(input, target, neg_ratio=3.):
    pos_logits = input[target > 0]
    pos_labels = target[target > 0]
    neg_logits = input[target == 0]
    neg_labels = target[target == 0]

    # calculate hard example numbers
    pos_num = pos_logits.numel()
    neg_num = neg_logits.numel()
    if pos_num * neg_ratio < neg_num:
        neg_num = max(1, int(pos_num * neg_ratio))
    else:
        pos_num = max(1, int(neg_num / neg_ratio))
    
    # top-k lowest positive scores
    pos_logits, pos_indices = (-pos_logits).topk(pos_num)
    pos_logits = -pos_logits
    pos_labels = pos_labels[pos_indices]

    # top-k highest negative scores
    neg_logits, neg_indices = neg_logits.topk(neg_num)
    neg_labels = neg_labels[neg_indices]

    loss = binary_cross_entropy_with_logits(
        jt.concat([pos_logits, neg_logits]),
        jt.concat([pos_labels, neg_labels]),
        reduction='mean')
    
    return loss


def smooth_l1_loss(input, target, beta=1. / 9):
    o = jt.abs(input - target)
    loss = jt.where(
        o < beta,
        0.5 * o ** 2 / beta,
        o - 0.5 * beta)
    return loss.mean()


def iou_loss(input, target, weight=None):
    # assume the order of [x1, y1, x2, y2]
    il, ir, it, ib = input.t()
    tl, tr, tt, tb = target.t()

    input_area = (il + ir) * (it + ib)
    target_area = (tl + tr) * (tt + tb)

    inter_w = jt.min(il, tl) + jt.min(ir, tr)
    inter_h = jt.min(ib, tb) + jt.min(it, tt)
    
    inter_area = inter_w * inter_h
    union_area = input_area + target_area - inter_area

    loss = -jt.log((inter_area + 1.0) / (union_area + 1.0))

    if weight is not None and weight.sum() > 0:
        return (loss * weight).sum() / weight.sum()
    else:
        return loss.mean()


def ghmr_loss(input, target, mu=0.02, bins=10, momentum=0):
    edges = [x / bins for x in range(bins + 1)]
    edges[-1] = 1e3
    mmt = momentum
    if mmt > 0:
        acc_sum = [0.0 for _ in range(bins)]

    # ASL1 loss
    diff = input - target
    loss = jt.sqrt(diff * diff + mu * mu) - mu

    # gradient length
    g = jt.abs(diff / jt.sqrt(mu * mu + diff * diff)).detach()
    weight = jt.zeros_like(g)

    tot = input.numel()
    n = 0
    for i in range(bins):
        inds = (g >= edges[i]) & (g < edges[i + 1])
        num_in_bin = inds.sum().item()
        if num_in_bin > 0:
            n += 1
            if mmt > 0:
                acc_sum[i] = mmt * acc_sum[i] \
                    + (1 - mmt) * num_in_bin
                weight[inds] = tot / acc_sum[i]
            else:
                weight[inds] = tot / num_in_bin
    if n > 0:
        weight /= n

    loss = loss * weight
    loss = loss.sum() / tot

    return loss


def label_smooth_loss(input, target, num_classes, eps=0.1):
    log_probs = nn.log_softmax(input, dim=1)
    target = log_probs.new_zeros(log_probs.size()).scatter_(
        1, target.unsqueeze(1), 1)
    target = (1 - eps) * target + eps / num_classes
    loss = (-target * log_probs).mean(dim=0).sum()
    return loss
