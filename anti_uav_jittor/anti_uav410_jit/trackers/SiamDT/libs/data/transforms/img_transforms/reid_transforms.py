import numpy as np
import jittor as jt

import libs.ops as ops
from libs.config import registry


__all__ = ['ReID_Transforms']


class Compose(object):
    
    def __init__(self, transforms):
        self.transforms = transforms
    
    def __call__(self, img, target):
        for t in self.transforms:
            img, target = t(img, target)
        return img, target
    
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


class CropAndResize(object):

    def __init__(self,
                 train=True,
                 context=0.5,
                 out_size=(128, 128),
                 random_shift=10):
        self.train = train
        self.context = context
        self.out_size = ops.make_pair(out_size)
        self.random_shift = random_shift

    def __call__(self, img, target):
        bbox = target['bbox']
        f_out_size = np.array(self.out_size, dtype=np.float32)

        # calculate the cropping size
        bbox_size = bbox[2:] - bbox[:2] + 1
        context = self.context * np.sum(bbox_size)
        crop_size = np.sqrt(np.prod(bbox_size + context))

        # calculate the cropping center
        bbox_center = (bbox[:2] + bbox[2:]) / 2.
        if self.train:
            # randomly shift in the cropped image
            shift = np.random.uniform(
                -self.random_shift, self.random_shift, 2)
            shift *= crop_size / f_out_size
        else:
            shift = np.zeros(2)
        crop_center = bbox_center + shift

        # crop and resize
        avg_color = np.mean(img, axis=(0, 1))
        crop_img = ops.crop_and_resize(
            img, crop_center, crop_size,
            out_size=self.out_size, border_value=avg_color)
        
        # bounding box in the cropped image
        scale = f_out_size / crop_size
        bbox_center = (f_out_size - 1) / 2. - shift * scale
        bbox_size *= scale
        crop_bbox = np.concatenate([
            bbox_center - (bbox_size - 1) / 2.,
            bbox_center + (bbox_size - 1) / 2.])
        target.update({'bbox': crop_bbox.astype(np.float32)})

        return crop_img, target


class RandomHorizontalFlip(object):

    def __init__(self, p=0.5):
        self.p = p
    
    def __call__(self, img, target):
        if np.random.rand() < self.p:
            img, target['bbox'] = ops.flip_img(img, target['bbox'])
        return img, target


class Normalize(object):

    def __init__(self,
                 rgb_mean=[0.485, 0.456, 0.406],
                 rgb_std=[0.229, 0.224, 0.225]):
        self.rgb_mean = rgb_mean
        self.rgb_std = rgb_std

    def __call__(self, img, target):
        img = ops.normalize_img(
            img / 255., self.rgb_mean, self.rgb_std)
        return img, target


class RandomErasing(object):
    
    def __init__(self,
                 p=0.5,
                 min_area=0.02,
                 max_area=0.4,
                 min_aspect_ratio=0.3,
                 rgb_mean=[0.485, 0.456, 0.406]):
        self.p = p
        self.min_area = min_area
        self.max_area = max_area
        self.min_aspect_ratio = min_aspect_ratio
        self.rgb_mean = rgb_mean
    
    def __call__(self, img, target):
        if np.random.rand() > self.p:
            return img, target
        
        h, w, c = img.shape
        for _ in range(100):
            area = np.prod(img.shape[:2])

            erase_area = area * np.random.uniform(
                self.min_area, self.max_area)
            erase_aspect_ratio = np.random.uniform(
                self.min_aspect_ratio, 1. / self.min_aspect_ratio)
            erase_h = int(round(np.sqrt(
                erase_area * erase_aspect_ratio)))
            erase_w = int(round(np.sqrt(
                erase_area / erase_aspect_ratio)))
            
            if erase_w < w and erase_h < h:
                x1 = np.random.randint(0, w - erase_w)
                y1 = np.random.randint(0, h - erase_h)
                img[y1:y1 + erase_h, x1:x1 + erase_w] = self.rgb_mean
                return img, target
        
        return img, target


class ToTensor(object):

    def __call__(self, img, target):
        img = jt.Var(img).permute(2, 0, 1).float()
        bbox = jt.Var(target['bbox']).float()
        label = jt.Var([target['ins_id']])[0] - 1
        target = {'bbox': bbox, 'label': label}
        return img, target


@registry.register_module
class ReID_Transforms(Compose):

    def __init__(self,
                 train=True,
                 context=0.5,
                 out_size=(128, 128),
                 random_shift=10,
                 flip_p=0.5,
                 rgb_mean=[0.485, 0.456, 0.406],
                 rgb_std=[0.229, 0.224, 0.225],
                 erase_p=0.5,
                 min_area=0.02,
                 max_area=0.4,
                 min_aspect_ratio=0.3):
        if train:
            super(ReID_Transforms, self).__init__(transforms=[
                CropAndResize(train, context, out_size, random_shift),
                RandomHorizontalFlip(flip_p),
                Normalize(rgb_mean, rgb_std),
                RandomErasing(erase_p, min_area, max_area,
                              min_aspect_ratio, rgb_mean),
                ToTensor()])
        else:
            super(ReID_Transforms, self).__init__(transforms=[
                CropAndResize(train, context, out_size, random_shift),
                Normalize(rgb_mean, rgb_std),
                ToTensor()])
