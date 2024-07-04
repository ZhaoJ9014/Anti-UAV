import numbers
import numpy as np
import cv2
import random
import math


def make_pair(p):
    if isinstance(p, numbers.Number):
        return (p, p)
    elif isinstance(p, (list, tuple)) and len(p) == 2:
        return p
    else:
        raise ValueError('Unable to convert {} to a pair'.format(p))


def bound_bboxes(bboxes, bound_size):
    w, h = bound_size
    bboxes[..., 0::2] = np.clip(bboxes[..., 0::2], 0, w - 1)
    bboxes[..., 1::2] = np.clip(bboxes[..., 1::2], 0, h - 1)
    if bboxes.shape[1] == 4:
        bboxes[..., 2:] = np.clip(
            bboxes[..., 2:], bboxes[..., :2], None)
    return bboxes


def random_interp():
    interps = [
        cv2.INTER_LINEAR,
        cv2.INTER_CUBIC,
        cv2.INTER_AREA,
        cv2.INTER_NEAREST,
        cv2.INTER_LANCZOS4]
    return np.random.choice(interps)


def rescale_img(img, scale, bboxes=None, interp=None):
    h, w = img.shape[:2]
    if isinstance(scale, (float, int)):
        assert scale > 0
        scale_factor = scale
    elif isinstance(scale, (list, tuple)):
        long_edge = max(scale)
        short_edge = min(scale)
        scale_factor = min(
            long_edge / max(h, w),
            short_edge / min(h, w))
    else:
        raise TypeError('Invalid type of scale')
    
    if interp is None:
        interp = random_interp()
    out_w = int(w * scale_factor + 0.5)
    out_h = int(h * scale_factor + 0.5)
    out_img = cv2.resize(img, (out_w, out_h), interpolation=interp)

    if bboxes is not None:
        out_bboxes = bboxes * scale_factor
        return out_img, out_bboxes, scale_factor
    else:
        return out_img, scale_factor


def resize_img(img, scale, bboxes=None, interp=None):
    if interp is None:
        interp = random_interp()
    h, w = img.shape[:2]
    out_w, out_h = scale
    out_img = cv2.resize(img, (out_w, out_h), interpolation=interp)
    scale_factor = (out_w / w, out_h / h)

    if bboxes is not None:
        out_bboxes = bboxes.copy()
        out_bboxes[..., 0::2] *= scale_factor[0]
        out_bboxes[..., 1::2] *= scale_factor[1]
        return out_img, out_bboxes, scale_factor
    else:
        return out_img, scale_factor


def normalize_img(img, mean, std):
    img = img.astype(np.float32)
    img -= mean
    img /= std
    return img


def denormalize_img(img, mean, std):
    img *= std
    img += mean
    return img


def stretch_color(img):
    img = img.astype(np.float32)
    return (img - img.min()) / (img.max() - img.min())


def flip_img(img, bboxes=None):
    img = np.flip(img, axis=1)
    if bboxes is None:
        return img
    else:
        w = img.shape[1]
        out_bboxes = bboxes.copy()
        out_bboxes[..., 0::4] = w - bboxes[..., 2::4] - 1
        out_bboxes[..., 2::4] = w - bboxes[..., 0::4] - 1
        return img, out_bboxes


def pad_img(img, shape, border_value=0):
    if len(shape) < len(img.shape):
        shape += (img.shape[-1], )
    assert all([so >= si for so, si in zip(shape, img.shape)])
    out_img = np.empty(shape, dtype=img.dtype)
    out_img[...] = border_value
    out_img[:img.shape[0], :img.shape[1], ...] = img
    return out_img


def pad_to_divisor(img, divisor, border_value=0):
    out_h = int(np.ceil(img.shape[0] / divisor) * divisor)
    out_w = int(np.ceil(img.shape[1] / divisor) * divisor)
    return pad_img(img, (out_h, out_w), border_value)


def bound_bboxes(bboxes, img_size):
    w, h = img_size
    bboxes[:, 0::2] = np.clip(bboxes[:, 0::2], 0, w - 1)
    bboxes[:, 1::2] = np.clip(bboxes[:, 1::2], 0, h - 1)
    if bboxes.shape[1] == 4:
        bboxes[:, 2:] = np.clip(bboxes[:, 2:], bboxes[:, :2], None)
    return bboxes


def jaccard(a, b):
    # a: N x 4 -> N x M x 4
    # b: M x 4 -> N x M x 4
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    area_i = (rb - lt).clip(min=0.).prod(axis=2)  # N x M

    area_a = (a[:, 2:] - a[:, :2]).prod(axis=1)  # N
    area_b = (b[:, 2:] - b[:, :2]).prod(axis=1)  # M
    ious = area_i / (
        area_a[:, None] + area_b[None, :] - area_i).clip(min=1e-6)
    
    return ious


def photometric_distort(img, swap_channels=False):
    # convert to float
    img = img.astype(np.float32)

    # random brightness
    if np.random.randint(2):
        delta = np.random.uniform(-32, 32)
        img += delta

    # order of random constrast
    constrast_first = np.random.randint(2)
    if constrast_first and np.random.randint(2):
        alpha = np.random.uniform(0.5, 1.5)
        img *= alpha
    
    # convert from RGB to HSV
    img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    # random Saturation
    if np.random.randint(2):
        alpha = np.random.uniform(0.5, 1.5)
        img[:, :, 1] *= alpha
    
    # random Hue
    if np.random.randint(2):
        delta = np.random.uniform(-18, 18)
        img[:, :, 0] += delta
        img[:, :, 0][img[:, :, 0] > 360.0] -= 360.0
        img[:, :, 0][img[:, :, 0] < 0.0] += 360.0
    
    # convert from HSV to RGB
    img = cv2.cvtColor(img, cv2.COLOR_HSV2RGB)

    # order of random constrast
    if not constrast_first and np.random.randint(2):
        alpha = np.random.uniform(0.5, 1.5)
        img *= alpha

    # randomly swap channels
    if swap_channels and np.random.randint(2):
        order = np.random.permutation(3)
        img = img[..., order]
    
    return img


def random_expand(img, bboxes, mean, min_ratio, max_ratio):
    if np.random.randint(2):
        return img, bboxes
    
    h, w, c = img.shape
    ratio = np.random.uniform(min_ratio, max_ratio)
    out_h, out_w = int(h * ratio), int(w * ratio)
    out_img = np.empty((out_h, out_w, c), dtype=img.dtype)
    out_img[...] = mean

    l = int(np.random.uniform(0, out_w - w))
    t = int(np.random.uniform(0, out_h - h))
    out_img[t:t + h, l:l + w, :] = img
    out_bboxes = bboxes + np.tile((l, t), 2)

    return out_img, out_bboxes


def random_crop(img, bboxes, min_ious, min_scale):
    h, w = img.shape[:2]
    min_ious = (1, *min_ious, 0)

    while True:
        min_iou = np.random.choice(min_ious)
        if min_iou == 1:
            valid_mask = np.ones(len(bboxes), dtype=bool)
            return img, bboxes, valid_mask
        
        for i in range(50):
            new_w = np.random.uniform(min_scale * w, w)
            new_h = np.random.uniform(min_scale * h, h)

            # aspect ratio should be between 0.5 and 2.0
            if new_h / new_w < 0.5 or new_h / new_w > 2.0:
                continue
            
            l = np.random.uniform(0, w - new_w)
            t = np.random.uniform(0, h - new_h)
            bound = np.array([l, t, l + new_w, t + new_h], dtype=int)

            # minimum IOU should be larger than min_iou
            ious = jaccard(
                bboxes.reshape(-1, 4),
                bound.reshape(-1, 4)).reshape(-1)
            if ious.min() < min_iou:
                continue
            
            # keep only bboxes with their centers inside the image
            centers = (bboxes[:, :2] + bboxes[:, 2:]) / 2.
            valid_mask = (bound[0] < centers[:, 0]) * \
                (bound[1] < centers[:, 1]) * \
                (bound[2] > centers[:, 0]) * \
                (bound[3] > centers[:, 1])
            if not valid_mask.any():
                continue

            # crop image and bboxes
            img = img[bound[1]:bound[3], bound[0]:bound[2], :]
            bboxes = bboxes.copy()
            bboxes[:, 2:] = bboxes[:, 2:].clip(max=bound[2:])
            bboxes[:, :2] = bboxes[:, :2].clip(min=bound[:2])
            bboxes -= np.tile(bound[:2], 2)

            return img, bboxes, valid_mask


def blend_(alpha, img1, img2):
    img1 *= alpha
    img2 *= (1 - alpha)
    img1 += img2


def brightness_(rng, img, gray, gray_mean, var):
    alpha = 1. + rng.uniform(low=-var, high=var)
    img *= alpha


def constrast_(rng, img, gray, gray_mean, var):
    alpha = 1. + rng.uniform(low=-var, high=var)
    blend_(alpha, img, gray_mean)


def saturation_(rng, img, gray, gray_mean, var):
    alpha = 1. + rng.uniform(low=-var, high=var)
    blend_(alpha, img, gray[:, :, None])


def lighting_(rng, img, eig_val, eig_vec, std):
    alpha = rng.normal(scale=std, size=(3, ))
    img += np.dot(eig_vec, eig_val * alpha)


def color_jitter(img, rng, eig_val, eig_vec, var=0.4, std=0.1):
    augs = [brightness_, constrast_, saturation_]
    random.shuffle(augs)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_mean = gray.mean()
    for f in augs:
        f(rng, img, gray, gray_mean, var=var)
    lighting_(rng, img, eig_val, eig_vec, std=std)

    return img


def _get_direction(point, rad):
    sin, cos = np.sin(rad), np.cos(rad)

    out = [0, 0]
    out[0] = point[0] * cos - point[1] * sin
    out[1] = point[0] * sin + point[1] * cos

    return out


def _get_3rd_point(a, b):
    direct = a - b
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)


def get_affine_transform(center, scale, rotation, output_size,
                         shift=None, inv=False):
    if not isinstance(scale, (list, tuple, np.ndarray)):
        scale = np.array([scale, scale], dtype=np.float32)
    if shift is None:
        shift = np.array([0, 0], dtype=np.float32)
    
    src_w = scale[0]
    dst_w = output_size[0]
    dst_h = output_size[1]

    rad = np.pi * rotation / 180
    src_dir = _get_direction([0, src_w * -0.5], rad)
    dst_dir = np.array([0, dst_w * -0.5], np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center + scale * shift
    src[1, :] = center + src_dir + scale * shift
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5], np.float32) + dst_dir

    src[2:, :] = _get_3rd_point(src[0, :], src[1, :])
    dst[2:, :] = _get_3rd_point(dst[0, :], dst[1, :])

    if not inv:
        return cv2.getAffineTransform(src, dst)
    else:
        return cv2.getAffineTransform(dst, src)


def apply_affine_transform(point, trans):
    out = np.array([point[0], point[1], 1.], dtype=np.float32).T
    out = np.dot(trans, out)
    return out[:2]


def random_resize(img, max_scale):
    interp = random_interp()
    scale = np.random.uniform(1. / max_scale, max_scale)
    out_size = (
        round(img.shape[1] * scale),
        round(img.shape[0] * scale))
    return cv2.resize(img, out_size, interpolation=interp)


def center_crop(img, size):
    h, w = img.shape[:2]
    tw, th = make_pair(size)
    i = round((h - th) / 2.)
    j = round((w - tw) / 2.)

    npad = max(0, -i, -j)
    if npad > 0:
        avg_color = np.mean(img, axis=(0, 1))
        img = cv2.copyMakeBorder(
            img, npad, npad, npad, npad,
            cv2.BORDER_CONSTANT, value=avg_color)
        i += npad
        j += npad

    return img[i:i + th, j:j + tw]

def swin_center_crop(img, size, bbox):
    h, w = img.shape[:2]
    th, tw = make_pair(size)
    i = round((h - th) / 2.)
    j = round((w - tw) / 2.)
    

    npad = max(0, -i, -j)
    if npad > 0:
        avg_color = np.mean(img, axis=(0, 1))
        img = cv2.copyMakeBorder(
            img, npad, npad, npad, npad,
            cv2.BORDER_CONSTANT, value=avg_color)
        i += npad
        j += npad

    newbbox=[bbox[0]-h/2+th/2, bbox[1]-w/2+tw/2,
             bbox[2]-h/2+th/2, bbox[3]-w/2+tw/2]

    return img[i:i + th, j:j + tw], newbbox


def simple_random_crop(img, size):
    h, w = img.shape[:2]
    tw, th = make_pair(size)
    i = np.random.randint(0, h - th + 1)
    j = np.random.randint(0, w - tw + 1)
    return img[i:i + th, j:j + tw]


def crop_and_resize(img, center, size, out_size,
                    border_type=cv2.BORDER_CONSTANT,
                    border_value=(0, 0, 0),
                    interp=cv2.INTER_LINEAR):
    # convert bbox to corners (0-indexed)
    size = np.round(size)
    corners = np.concatenate((
        np.round(center - (size - 1) / 2),
        np.round(center - (size - 1) / 2) + size))
    corners = np.round(corners).astype(int)

    # pad image if necessary
    pads = np.concatenate((
        -corners[:2], corners[2:] - img.shape[1::-1]))
    npad = max(0, int(pads.max()))
    if npad > 0:
        img = cv2.copyMakeBorder(
            img, npad, npad, npad, npad,
            border_type, value=border_value)

    # crop image patch
    corners = (corners + npad).astype(int)
    patch = img[corners[1]:corners[3], corners[0]:corners[2]]

    # resize to out_size
    out_size = make_pair(out_size)
    patch = cv2.resize(patch, out_size, interpolation=interp)

    return patch


def swin_crop_and_resize(img, center, size, out_size,
                    border_type=cv2.BORDER_CONSTANT,
                    border_value=(0, 0, 0),
                    interp=cv2.INTER_LINEAR):
    # convert bbox to corners (0-indexed)
    size = np.round(size)
    corners = np.concatenate((
        np.round(center - (size - 1) / 2),
        np.round(center - (size - 1) / 2) + size))
    corners = np.round(corners).astype(int)

    # pad image if necessary
    pads = np.concatenate((
        -corners[:2], corners[2:] - img.shape[1::-1]))
    npad = max(0, int(pads.max()))
    if npad > 0:
        img = cv2.copyMakeBorder(
            img, npad, npad, npad, npad,
            border_type, value=border_value)

    # crop image patch
    corners = (corners + npad).astype(int)
    patch = img[corners[1]:corners[3], corners[0]:corners[2]]

    src_size=patch.shape[0]
    ratio=out_size/src_size

    # resize to out_size
    out_size = make_pair(out_size)
    patch = cv2.resize(patch, out_size, interpolation=interp)
    
    return patch, ratio


def crop_square(img, bbox, context, src_size, dst_size):
    center = (bbox[:2] + bbox[2:]) / 2.
    target_sz = bbox[2:] - bbox[:2] + 1

    context = context * np.sum(target_sz)
    size = np.sqrt(np.prod(target_sz + context))
    size *= dst_size / src_size

    avg_color = np.mean(img, axis=(0, 1), dtype=float)
    interp = random_interp()
    patch = crop_and_resize(
        img, center, size, dst_size,
        border_value=avg_color, interp=interp)
    
    return patch

def swin_crop_square(img, bbox, search_area_scale, dst_size):
    center = (bbox[:2] + bbox[2:]) / 2.
    target_sz = bbox[2:] - bbox[:2] + 1

    length=math.sqrt(target_sz[0]**2 + target_sz[1]**2)

    size = length * search_area_scale # ignores target aspect ratio

    avg_color = np.mean(img, axis=(0, 1), dtype=float)
    interp = random_interp()
    patch, ratio = swin_crop_and_resize(
        img, center, size, dst_size,
        border_value=avg_color, interp=interp)
    
    newtarget_sz=target_sz*ratio
    newbbox=[dst_size/2-(newtarget_sz[0]-1)/2, dst_size/2-(newtarget_sz[1]-1)/2,
             dst_size/2+(newtarget_sz[0]-1)/2, dst_size/2+(newtarget_sz[1]-1)/2]
    
    return patch, newbbox, ratio
