import numpy as np
import math
import jittor as jt
# import torch.nn.functional as F
import cv2 as cv
import random


class Transform:
    """Base data augmentation transform class."""

    def __init__(self, output_sz = None, shift = None):
        self.output_sz = output_sz
        self.shift = (0,0) if shift is None else shift

    def __call__(self, image, is_mask=False):
        raise NotImplementedError

    def crop_to_output(self, image):
        if isinstance(image, jt.Var):
            imsz = image.shape[2:]
            if self.output_sz is None:
                pad_h = 0
                pad_w = 0
            else:
                pad_h = (self.output_sz[0] - imsz[0]) / 2
                pad_w = (self.output_sz[1] - imsz[1]) / 2

            pad_left = math.floor(pad_w) + self.shift[1]
            pad_right = math.ceil(pad_w) - self.shift[1]
            pad_top = math.floor(pad_h) + self.shift[0]
            pad_bottom = math.ceil(pad_h) - self.shift[0]

            return jt.nn.pad(image, (pad_left, pad_right, pad_top, pad_bottom), 'replicate')
        else:
            raise NotImplementedError

class Blur(Transform):
    """Blur with given sigma (can be axis dependent)."""
    def __init__(self, sigma, output_sz = None, shift = None):
        super().__init__(output_sz, shift)
        if isinstance(sigma, (float, int)):
            sigma = (sigma, sigma)
        self.sigma = sigma
        self.filter_size = [math.ceil(2*s) for s in self.sigma]
        x_coord = [jt.arange(-sz, sz+1, dtype=jt.float32) for sz in self.filter_size]
        self.filter = [jt.exp(-(x**2)/(2*s**2)) for x, s in zip(x_coord, self.sigma)]
        self.filter[0] = self.filter[0].view(1,1,-1,1) / self.filter[0].sum()
        self.filter[1] = self.filter[1].view(1,1,1,-1) / self.filter[1].sum()

    def __call__(self, image, is_mask=False):
        if isinstance(image, jt.Var):
            sz = image.shape[2:]
            im1 = jt.nn.conv2d(image.view(-1,1,sz[0],sz[1]), self.filter[0], padding=(self.filter_size[0],0))
            return self.crop_to_output(jt.nn.conv2d(im1, self.filter[1], padding=(0,self.filter_size[1])).view(1,-1,sz[0],sz[1]))
        else:
            raise NotImplementedError


class FlipHorizontal(Transform):
    """Flip along horizontal axis."""
    def __call__(self, image, is_mask=False):
        if isinstance(image, jt.Var):
            return self.crop_to_output(image.flip((3,)))
        else:
            return np.fliplr(image)

class FlipVertical(Transform):
    """Flip along vertical axis."""
    def __call__(self, image: jt.Var, is_mask=False):
        if isinstance(image, jt.Var):
            return self.crop_to_output(image.flip((2,)))
        else:
            return np.flipud(image)