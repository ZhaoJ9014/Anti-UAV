import torch
import jittor as jt
from pytracking.features.featurebase import FeatureBase


class RGB(FeatureBase):
    """RGB feature normalized to [-0.5, 0.5]."""
    def dim(self):
        return 3

    def stride(self):
        return self.pool_stride

    def extract(self, im: jt.Var):
        return im/255 - 0.5


class Grayscale(FeatureBase):
    """Grayscale feature normalized to [-0.5, 0.5]."""
    def dim(self):
        return 1

    def stride(self):
        return self.pool_stride

    def extract(self, im: jt.Var):
        return jt.mean(im/255 - 0.5, 1, keepdim=True)
