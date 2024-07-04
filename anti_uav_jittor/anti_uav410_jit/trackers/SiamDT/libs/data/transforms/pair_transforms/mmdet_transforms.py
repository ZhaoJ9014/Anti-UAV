import numpy as np
import jittor as jt
import copy
import cv2

import libs.ops as ops
from libs.config import registry


__all__ = ['BasicPairTransforms', 'ExtraPairTransforms', 'TTFNetTransforms']


class _PairTransform(object):

    def __call__(self, item):
        # process query
        item['img_z'], item['img_meta_z'], item['gt_bboxes_z'] = \
            self._process_query(
                item['img_z'],
                item['img_meta_z'],
                item['gt_bboxes_z'])
        
        # process gallary
        item['img_x'], item['img_meta_x'], item['gt_bboxes_x'] = \
            self._process_gallary(
                item['img_x'],
                item['img_meta_x'],
                item['gt_bboxes_x'])
        
        return item
    
    def _process_query(self, img, meta, bboxes=None):
        raise NotImplementedError
    
    def _process_gallary(self, img, meta, bboxes=None):
        raise NotImplementedError


class Compose(_PairTransform):

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, item):
        for t in self.transforms:
            item = t(item)
        return item

    def __getitem__(self, index):
        if isinstance(index, int):
            return self.transforms[index]
        elif isinstance(index, slice):
            return Compose(self.transforms[index])
        else:
            raise TypeError('Invalid type of index.')

    def __add__(self, other):
        if isinstance(other, Compose):
            return Compose(self.transforms + other.transforms)
        else:
            raise TypeError('Invalid type of other.')

    def _process_query(self, img, meta, bboxes=None):
        for t in self.transforms:
            img, meta, bboxes = t._process_query(img, meta, bboxes)
        return img, meta, bboxes

    def _process_gallary(self, img, meta, bboxes=None):
        for t in self.transforms:
            img, meta, bboxes = t._process_gallary(img, meta, bboxes)
        return img, meta, bboxes


class Rescale(_PairTransform):

    def __init__(self,
                 scale=(1333, 800),
                 interp=None):
        self.scale = scale
        self.interp = interp
        self._process_query = self._process
        self._process_gallary = self._process

    def _process(self, img, meta, bboxes=None):
        if bboxes is None:
            img, scale_factor = ops.rescale_img(
                img, self.scale, bboxes=None, interp=self.interp)
        else:
            img, bboxes, scale_factor = ops.rescale_img(
                img, self.scale, bboxes=bboxes, interp=self.interp)
        meta.update({
            'img_shape': img.shape,
            'scale_factor': scale_factor})
        return img, meta, bboxes


class Resize(_PairTransform):

    def __init__(self,
                 scale=(512, 512),
                 interp=None):
        self.scale = scale
        self.interp = interp
        self._process_query = self._process
        self._process_gallary = self._process
    
    def _process(self, img, meta, bboxes=None):
        if bboxes is None:
            img, scale_factor = ops.resize_img(
                img, self.scale, bboxes=None, interp=self.interp)
        else:
            img, bboxes, scale_factor = ops.resize_img(
                img, self.scale, bboxes=bboxes, interp=self.interp)
        meta.update({
            'img_shape': img.shape,
            'scale_factor': scale_factor})
        return img, meta, bboxes


class Normalize(_PairTransform):

    def __init__(self,
                 mean=[123.675, 116.28, 103.53],
                 std=[58.395, 57.12, 57.375]):
        self.mean = mean
        self.std = std
        self._process_query = self._process
        self._process_gallary = self._process

    def _process(self, img, meta, bboxes=None):
        img = ops.normalize_img(img, self.mean, self.std)
        return img, meta, bboxes


class RandomFlip(_PairTransform):

    def __init__(self, p=0.5):
        self.p = p
        self._process_query = self._process
        self._process_gallary = self._process

    def _process(self, img, meta, bboxes=None):
        if np.random.rand() < self.p:
            if bboxes is None:
                img = ops.flip_img(img)
            else:
                img, bboxes = ops.flip_img(img, bboxes)
            meta.update({'flip': True})
        else:
            meta.update({'flip': False})
        return img, meta, bboxes


class PadToDivisor(_PairTransform):

    def __init__(self, divisor=32, border_value=0):
        self.divisor = divisor
        self.border_value = border_value
        self._process_query = self._process
        self._process_gallary = self._process

    def _process(self, img, meta, bboxes=None):
        img = ops.pad_to_divisor(img, self.divisor, self.border_value)
        meta.update({'pad_shape': img.shape})
        return img, meta, bboxes


class BoundBoxes(_PairTransform):

    def __init__(self):
        self._process_query = self._process
        self._process_gallary = self._process

    def _process(self, img, meta, bboxes=None):
        if bboxes is not None:
            bboxes = ops.bound_bboxes(bboxes, img.shape[1::-1])
        return img, meta, bboxes


class ToTensor(_PairTransform):

    def __init__(self):
        self._process_query = self._process
        self._process_gallary = self._process

    def _process(self, img, meta, bboxes=None):
        img = jt.Var(img.transpose(2, 0, 1)).float()
        if bboxes is not None:
            bboxes = jt.Var(bboxes).float()
        return img, meta, bboxes


class PhotometricDistort(_PairTransform):

    def __init__(self, swap_channels=True):
        self.swap_channels = swap_channels
        self._swap_order = None

    def _process_query(self, img, meta, bboxes=None):
        img = ops.photometric_distort(img, swap_channels=False)
        if self.swap_channels and np.random.randint(2):
            order = np.random.permutation(3)
            img = img[..., order]
            self._swap_order = order
        else:
            self._swap_order = None
        return img, meta, bboxes

    def _process_gallary(self, img, meta, bboxes=None):
        img = ops.photometric_distort(img, swap_channels=False)
        if self.swap_channels and self._swap_order is not None:
            img = img[..., self._swap_order]
        return img, meta, bboxes


class RandomExpand(_PairTransform):

    def __init__(self,
                 mean=[123.675, 116.28, 103.53],
                 min_ratio=1,
                 max_ratio=4):
        self.mean = mean
        self.min_ratio = min_ratio
        self.max_ratio = max_ratio
        self._process_query = self._process
        self._process_gallary = self._process

    def _process(self, img, meta, bboxes):
        if bboxes is None:
            raise ValueError('Unsupport None type of bboxes')
        img, bboxes = ops.random_expand(
            img, bboxes, self.mean, self.min_ratio, self.max_ratio)
        return img, meta, bboxes


class RandomCrop(_PairTransform):

    def __init__(self,
                 min_ious=[0.1, 0.3, 0.5, 0.7, 0.9],
                 min_scale=0.3):
        self.min_ious = min_ious
        self.min_scale = min_scale

    def __call__(self, item):
        item0 = item
        for i in range(10):
            item = copy.deepcopy(item0)
            item['img_z'], item['gt_bboxes_z'], mask_z = ops.random_crop(
                item['img_z'], item['gt_bboxes_z'],
                self.min_ious, self.min_scale)
            item['img_x'], item['gt_bboxes_x'], mask_x = ops.random_crop(
                item['img_x'], item['gt_bboxes_x'],
                self.min_ious, self.min_scale)
            mask = mask_z & mask_x
            if not mask.any():
                continue
            item['gt_bboxes_z'] = item['gt_bboxes_z'][mask]
            item['gt_bboxes_x'] = item['gt_bboxes_x'][mask]
            return item
        return item0

    def _process_query(self, img, meta, bboxes):
        raise NotImplementedError(
            'Separately processing query is not supported')

    def _process_gallary(self, img, meta, bboxes):
        raise NotImplementedError(
            'Separately processing gallary is not supported')


@registry.register_module
class BasicPairTransforms(Compose):

    def __init__(self,
                 train=True,
                 scale=(1333, 800),
                 mean=[123.675, 116.28, 103.53],
                 std=[58.395, 57.12, 57.375],
                 interp=None,
                 flip_p=0.5,
                 pad_divisor=32,
                 border_value=0):
        if not train:
            flip_p = 0.
            interp = cv2.INTER_LINEAR if interp is None else interp
        super(BasicPairTransforms, self).__init__(transforms=[
            Rescale(scale, interp),
            Normalize(mean, std),
            RandomFlip(flip_p),
            PadToDivisor(pad_divisor, border_value),
            BoundBoxes(),
            ToTensor()])
    
    def __call__(self, *args):
        assert len(args) in [1, 3], 'Invalid number of arguments'
        if len(args) == 3:
            img_z, img_x, target = args
            item = {
                'img_z': img_z,
                'img_x': img_x,
                'img_meta_z': {'ori_shape': img_z.shape},
                'img_meta_x': {'ori_shape': img_x.shape},
                'gt_bboxes_z': target['bboxes_z'],
                'gt_bboxes_x': target['bboxes_x']}
            return super(BasicPairTransforms, self).__call__(item)
        else:
            return super(BasicPairTransforms, self).__call__(args[0])


@registry.register_module
class ExtraPairTransforms(Compose):

    def __init__(self,
                 with_photometric=True,
                 with_expand=True,
                 with_crop=True,
                 with_basic=True,
                 swap_channels=True,
                 mean=[123.675, 116.28, 103.53],
                 min_ratio=1,
                 max_ratio=4,
                 min_ious=[0.1, 0.3, 0.5, 0.7, 0.9],
                 min_scale=0.3,
                 **kwargs):
        transforms = []
        if with_photometric:
            transforms += [PhotometricDistort(swap_channels)]
        if with_expand:
            transforms += [RandomExpand(mean, min_ratio, max_ratio)]
        if with_crop:
            transforms += [RandomCrop(min_ious, min_scale)]
        if with_basic:
            transforms += [BasicPairTransforms(**kwargs)]
        super(ExtraPairTransforms, self).__init__(transforms)
    
    def __call__(self, *args):
        assert len(args) in [1, 3], 'Invalid number of arguments'
        if len(args) == 3:
            img_z, img_x, target = args
            item = {
                'img_z': img_z,
                'img_x': img_x,
                'img_meta_z': {'ori_shape': img_z.shape},
                'img_meta_x': {'ori_shape': img_x.shape},
                'gt_bboxes_z': target['bboxes_z'],
                'gt_bboxes_x': target['bboxes_x']}
            return super(ExtraPairTransforms, self).__call__(item)
        else:
            return super(ExtraPairTransforms, self).__call__(args[0])


@registry.register_module
class TTFNetTransforms(Compose):

    def __init__(self,
                 train=True,
                 scale=(512, 512),
                 mean=[123.675, 116.28, 103.53],
                 std=[58.395, 57.12, 57.375],
                 interp=None,
                 flip_p=0.5,
                 pad_divisor=32,
                 border_value=0,
                 swap_channels=True):
        if train:
            transforms = [PhotometricDistort(swap_channels)]
        else:
            flip_p = 0.
            interp = cv2.INTER_LINEAR if interp is None else interp
            transforms = []
        transforms += [
            Resize(scale, interp),
            Normalize(mean, std),
            RandomFlip(flip_p),
            PadToDivisor(pad_divisor, border_value),
            BoundBoxes(),
            ToTensor()]
        super(TTFNetTransforms, self).__init__(transforms)
    
    def __call__(self, *args):
        assert len(args) in [1, 3], 'Invalid number of arguments'
        if len(args) == 3:
            img_z, img_x, target = args
            item = {
                'img_z': img_z,
                'img_x': img_x,
                'img_meta_z': {'ori_shape': img_z.shape},
                'img_meta_x': {'ori_shape': img_x.shape},
                'gt_bboxes_z': target['bboxes_z'],
                'gt_bboxes_x': target['bboxes_x']}
            return super(TTFNetTransforms, self).__call__(item)
        else:
            return super(TTFNetTransforms, self).__call__(args[0])
