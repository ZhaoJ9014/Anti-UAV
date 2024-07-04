import numbers
import jittor as jt
# import torchvision.transforms as T
from Compose_util import *
import numpy as np

import libs.ops as ops
from libs.config import registry


__all__ = ['SiamFC_Transforms']


class RandomResize(object):

    def __init__(self, max_scale=1.05):
        self.max_scale = max_scale
    
    def __call__(self, img):
        return ops.random_resize(img, self.max_scale)


class CenterCrop(object):

    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
    
    def __call__(self, img):
        return ops.center_crop(img, self.size)


class RandomCrop(object):

    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
    
    def __call__(self, img):
        return ops.simple_random_crop(img, self.size)


class ToTensor(object):

    def __call__(self, img):
        return jt.Var(img).float().permute(2, 0, 1)


@registry.register_module
class SiamFC_Transforms(object):

    def __init__(self,
                 exemplar_sz=127,
                 instance_sz=255,
                 context=0.5,
                 shift=8,
                 out_stride=8,
                 response_sz=15,
                 r_pos=16,
                 r_neg=0):
        self.exemplar_sz = exemplar_sz
        self.instance_sz = instance_sz
        self.context = context
        self.shift = shift
        self.out_stride = out_stride
        self.response_sz = response_sz
        self.r_pos = r_pos
        self.r_neg = r_neg

        # transforms for the query image
        self.transforms_z = Compose([
            RandomResize(max_scale=1.05),
            CenterCrop(size=instance_sz - shift),
            RandomCrop(size=instance_sz - 2 * shift),
            CenterCrop(size=exemplar_sz),
            ToTensor()])
        
        # transforms for the search image
        self.transforms_x = Compose([
            RandomResize(max_scale=1.05),
            CenterCrop(instance_sz - shift),
            RandomCrop(instance_sz - 2 * shift),
            ToTensor()])
    
    def __call__(self, img_z, img_x, target):
        # crop image pair and perform data augmentation
        img_z = ops.crop_square(
            img_z, target['bboxes_z'][0],
            self.context, self.exemplar_sz, self.instance_sz)
        img_x = ops.crop_square(
            img_x, target['bboxes_x'][0],
            self.context, self.exemplar_sz, self.instance_sz)
        img_z = self.transforms_z(img_z)
        img_x = self.transforms_x(img_x)

        # build training target
        target = self._build_target(self.response_sz)

        return img_z, img_x, target
    
    def _build_target(self, response_sz):
        # skip if same sized labels has already been created
        response_sz = ops.make_pair(response_sz)
        if hasattr(self, '_labels') and self._labels.size() == response_sz:
            return self._labels

        def logistic_labels(x, y, r_pos, r_neg):
            dist = np.abs(x) + np.abs(y)  # block distance
            labels = np.where(dist <= r_pos,
                              np.ones_like(x),
                              np.where(dist < r_neg,
                                       np.ones_like(x) * 0.5,
                                       np.zeros_like(x)))
            return labels

        # distances along x- and y-axis
        h, w = response_sz
        x = np.arange(w) - (w - 1) / 2
        y = np.arange(h) - (h - 1) / 2
        x, y = np.meshgrid(x, y)

        # create logistic labels
        r_pos = self.r_pos / self.out_stride
        r_neg = self.r_neg / self.out_stride
        labels = logistic_labels(x, y, r_pos, r_neg)

        # convert to tensors (add the channel dimension)
        self._labels = jt.Var(labels).float().unsqueeze(0)
        
        return self._labels
