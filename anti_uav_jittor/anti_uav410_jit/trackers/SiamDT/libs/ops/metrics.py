import numpy as np
import jittor as jt
from shapely import geometry


def center_error(r1, r2):
    r1, r2 = np.broadcast_arrays(r1, r2)
    c1 = (r1[..., :2] + r1[..., 2:]) / 2.
    c2 = (r2[..., :2] + r2[..., 2:]) / 2.
    return np.sqrt(np.sum(np.power(c1 - c2, 2), axis=-1))


def rect_iou(r1, r2, bound=None):
    r1, r2 = np.broadcast_arrays(r1, r2)
    if bound is not None:
        w, h = bound
        r1[..., 0::2] = r1[..., 0::2].clip(0, w - 1)
        r1[..., 1::2] = r1[..., 1::2].clip(0, h - 1)
        r2[..., 0::2] = r2[..., 0::2].clip(0, w - 1)
        r2[..., 1::2] = r2[..., 1::2].clip(0, h - 1)

    # areas of r1 and r2
    r1[..., 2:] = r1[..., 2:].clip(r1[..., :2] - 1)
    r2[..., 2:] = r2[..., 2:].clip(r2[..., :2] - 1)
    r1_areas = np.prod(r1[..., 2:] - r1[..., :2] + 1, axis=-1)
    r2_areas = np.prod(r2[..., 2:] - r2[..., :2] + 1, axis=-1)

    # areas of intersection and union
    lt = np.maximum(r1[..., :2], r2[..., :2])
    rb = np.minimum(r1[..., 2:], r2[..., 2:]).clip(lt - 1)
    inter_areas = np.prod(rb - lt + 1, axis=-1)
    union_areas = r1_areas + r2_areas - inter_areas

    return inter_areas / union_areas.clip(1e-12)


def poly_iou(p1, p2, bound=None):
    def to_polygon(u):
        if u.shape[-1] == 4:
            return [geometry.box(p[0], p[1], p[2] + 1, p[3] + 1)
                    for p in u]
        elif u.shape[-1] == 8:
            return [geometry.Polygon([(p[2 * i], p[2 * i + 1])
                    for i in range(4)]) for p in u]
        else:
            raise ValueError('Expected the last dimension to be 4 or 8,'
                             'but got {}'.format(u.shape[-1]))

    # ensure p1 and p2 to be 2-dimensional
    if p1.ndim == 1:
        p1 = p1[np.newaxis, :]
    if p2.ndim == 1:
        p2 = p2[np.newaxis, :]
    assert p1.ndim == 2
    assert p2.ndim == 2

    # convert to Polygons
    p1 = to_polygon(p1)
    p2 = to_polygon(p2)
    
    # bound Polygons
    if bound is not None:
        bound = geometry.box(0, 0, bound[0], bound[1])
        p1 = [p.intersection(bound) for p in p1]
        p2 = [p.intersection(bound) for p in p2]
    
    # calculate IOUs
    ious = []
    for p1_, p2_ in zip(p1, p2):
        inter_area = p1_.intersection(p2_).area
        union_area = p1_.union(p2_).area
        ious.append(inter_area / max(union_area, 1e-12))
    
    return np.array(ious)


def euclidean(x, y, sqrt=True):
    m, n = x.size(0), y.size(0)
    x2 = jt.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    y2 = jt.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist_mat = x2 + y2
    dist_mat.addmm_(1, -2, x, y.t()).clamp_(min=1e-12)
    if sqrt:
        dist_mat = dist_mat.sqrt()
    return dist_mat


def precision_recall(scores, labels, thr=0.6):
    pred = scores.gt(thr)
    gt = labels.gt(0)

    tp = (pred & gt).sum().float()
    fp = (pred & ~gt).sum().float()
    fn = (~pred & gt).sum().float()

    precision = tp / (tp + fp).clamp_(1e-12)
    recall = tp / (tp + fn).clamp_(1e-12)
    f1_score = 2 * precision * recall / (
        precision + recall).clamp_(1e-12)
    
    metrics = {
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score}
    
    return metrics


def r1_map(dist_mat, labels):
    indices = jt.argsort(dist_mat, dim=1)
    matches = (labels[indices] == labels.unsqueeze(1)).float()

    n = dist_mat.size(0)
    cmc, ap = [], []
    for i in range(n):
        ins_id = labels[i]

        matches_i = matches[i][indices[i] != i]
        cmc_i = matches_i.cumsum(dim=0).clamp_(None, 1.)
        cmc.append(cmc_i)

        num_matches = matches_i.sum()
        tmp_cmc = matches_i.cumsum(dim=0) * matches_i
        tmp_cmc /= jt.arange(
            1, len(tmp_cmc) + 1, dtype=tmp_cmc.dtype,
            device=tmp_cmc.device)
        ap_i = tmp_cmc.sum() / num_matches
        ap.append(ap_i)
    
    cmc = jt.stack(cmc, dim=0).mean(dim=0)
    mean_ap = jt.stack(ap, dim=0).mean(dim=0)
    metrics = {
        'cmc_1': cmc[0],
        'cmc_2': cmc[1],
        'cmc_5': cmc[4],
        'cmc_10': cmc[9],
        'mean_ap': mean_ap}

    return metrics


def topk_precision(scores, labels):
    pred = jt.argsort(scores, dim=1, descending=True)
    matches = (pred == labels.unsqueeze(1))
    topk = [matches[:, :k].any(dim=1).float().mean()
            for k in range(1, matches.size(1) + 1)]
    output = {
        'top1': topk[0],
        'top5': topk[4],
        'top10': topk[9]}
    return output
