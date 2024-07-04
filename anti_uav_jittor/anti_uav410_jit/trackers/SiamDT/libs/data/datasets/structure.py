import numpy as np
from jittor.dataset import Dataset

from .dataset import PairDataset, InstanceDataset
from libs.config import registry
from libs import ops
from collections import OrderedDict


__all__ = ['Seq2Pair', 'Image2Pair', 'Seq2Instance', 'Subset', 'Slice',
           'RandomConcat']


@registry.register_module
class Seq2Pair(PairDataset):

    def __init__(self, seqs, transforms=None,
                 pairs_per_seq=10, max_distance=300):

        super(Seq2Pair, self).__init__(
            name='{}_pairs'.format(seqs.name))
        self.seqs = seqs
        self.transforms = transforms
        self.pairs_per_seq = pairs_per_seq
        self.max_distance = max_distance
        # group sequences by aspect ratios (for sampling)
        self.group_flags = self._build_group_flags(seqs, pairs_per_seq)
    
    def __getitem__(self, index):
        if index > len(self):
            raise IndexError('Index out of range')
        index %= len(self.seqs)

        # get image files and annotations
        img_files, target = self.seqs[index]
        anno, meta = target['anno'], target['meta']
        # filter noisy annotations
        valid_indices = self._filter(anno, meta)
        if len(valid_indices) < 2:
            return self._random_next()
        
        # ramdomly sample a pair
        rand_z, rand_x = self._sample_pair(anno, valid_indices)
        if rand_z < 0 or rand_x < 0:
            return self._random_next()

        # annotations pair
        if anno.shape[1] in [4, 8]:
            # SOT-based pairs
            bboxes_z = np.expand_dims(anno[rand_z], axis=0)
            bboxes_x = np.expand_dims(anno[rand_x], axis=0)
        elif anno.shape[1] == 9:
            # MOT/VID-based pairs
            # find consistent track_ids
            mask_z = (anno[:, 0] == rand_z)
            mask_x = (anno[:, 0] == rand_x)
            z_ids, x_ids = anno[mask_z, 1], anno[mask_x, 1]
            join_ids = sorted(set(z_ids) & set(x_ids))

            # extract bounding boxes (ignore the class label)
            bboxes_z, bboxes_x = [], []
            for track_id in join_ids:
                mask_id = (anno[:, 1] == track_id)
                mask_id_z = (mask_z & mask_id)
                mask_id_x = (mask_x & mask_id)
                if mask_id_z.sum() != 1 or mask_id_x.sum() != 1:
                    ops.sys_print('Warning: found repeated ID for',
                                  self.seqs.seq_names[index])
                    return self._random_next()
                bboxes_z += [anno[mask_id_z, 2:6]]
                bboxes_x += [anno[mask_id_x, 2:6]]
            bboxes_z = np.concatenate(bboxes_z, axis=0)
            bboxes_x = np.concatenate(bboxes_x, axis=0)
            assert len(bboxes_z) == len(bboxes_x) == len(join_ids)
        
        # image pair
        img_z = ops.read_image(img_files[rand_z])
        img_x = ops.read_image(img_files[rand_x])

        # bound annotations by image boundaries
        h, w = img_z.shape[:2]
        bboxes_z = ops.bound_bboxes(bboxes_z, [w, h])
        bboxes_x = ops.bound_bboxes(bboxes_x, [w, h])

        # build target
        target = {
            'bboxes_z': bboxes_z,
            'bboxes_x': bboxes_x}
        
        # apply transforms if applicable
        if self.transforms is not None:
            return self.transforms(img_z, img_x, target)
        else:
            return img_z, img_x, target

    def __len__(self):
        return len(self.seqs) * self.pairs_per_seq
    
    def _sample_pair(self, anno, indices):
        ndims = anno.shape[1]
        n = len(indices)
        assert ndims in [4, 8, 9]
        assert n > 0

        if ndims in [4, 8]:
            if n == 1:
                return indices[0], indices[0]
            elif n == 2:
                return indices[0], indices[1]
            else:
                for _ in range(100):
                    rand_z, rand_x = sorted(
                        np.random.choice(indices, 2, replace=False))
                    if rand_x - rand_z <= self.max_distance:
                        break
                else:
                    rand_z, rand_x = np.random.choice(indices, 2)
        elif ndims == 9:
            anno = anno[indices]
            track_ids, counts = np.unique(anno[:, 1], return_counts=True)
            if np.all(counts < 2):
                return -1, -1
            track_id = np.random.choice(track_ids[counts >= 2])
            frames = np.unique(anno[anno[:, 1] == track_id, 0]).astype(np.int64)
            rand_z, rand_x = np.random.choice(frames, 2)
        
        return rand_z, rand_x
    
    def _filter(self, anno, meta):
        ndims = anno.shape[1]
        assert ndims in [4, 8, 9]

        if ndims == 8:
            return np.arange(len(anno))
        elif ndims == 4:
            bboxes = anno.copy()
        elif ndims == 9:
            bboxes = anno[:, 2:6].copy()
        bboxes[:, 2:] = bboxes[:, 2:] - bboxes[:, :2] + 1
        size = np.array([[
            meta['width'], meta['height']]], dtype=np.float32)
        areas = bboxes[:, 2] * bboxes[:, 3]

        # conditions
        conditions = [
            areas > 20,
            np.all(bboxes[:, 2:] >= 10, axis=1),
            np.all(bboxes[:, 2:] <= 960, axis=1),
            np.all((bboxes[:, 2:] / size) >= 0.01, axis=1),
            np.all((bboxes[:, 2:] / size) <= 0.75, axis=1),
            bboxes[:, 2] / np.maximum(1, bboxes[:, 3]) >= 0.2,
            bboxes[:, 2] / np.maximum(1, bboxes[:, 3]) < 5]
        if ndims == 9:
            conditions.append(anno[:, 8] <= 0.5)
        if 'cover' in meta:
            cover = meta['cover']
            conditions.append(
                cover > max(1., cover.max() * 0.3))
        
        mask = np.logical_and.reduce(conditions)
        indices = np.where(mask)[0]

        return indices
    
    def _random_next(self):
        index = np.random.choice(len(self))
        return self.__getitem__(index)
    
    def _build_group_flags(self, seqs, pairs_per_seq):
        flags = np.zeros(len(seqs), dtype=np.uint8)
        for i, (_, target) in enumerate(seqs):
            meta = target['meta']
            if meta['width'] / meta['height'] > 1:
                flags[i] = 1
        flags = np.tile(flags, pairs_per_seq)
        assert len(flags) == len(self)
        return flags


@registry.register_module
class Image2Pair(PairDataset):
    
    def __init__(self, imgs, transforms=None):
        super(Image2Pair, self).__init__(
            name='{}_pairs'.format(imgs.name))
        self.imgs = imgs
        self.transforms = transforms
        # group images by aspect ratios (for sampling)
        self.group_flags = self._build_group_flags(imgs)
    
    def __getitem__(self, index):
        img, target = self.imgs[index]
        bboxes = target['bboxes']
        if len(bboxes) == 0:
            return self._random_next()
        
        img_z, bboxes_z = img, bboxes
        img_x, bboxes_x = img.copy(), bboxes.copy()

        # bound annotations by image boundaries
        h, w = img_z.shape[:2]
        bboxes_z = ops.bound_bboxes(bboxes_z, [w, h])
        bboxes_x = ops.bound_bboxes(bboxes_x, [w, h])

        # build target
        target = {
            'bboxes_z': bboxes_z,
            'bboxes_x': bboxes_x}
        
        # apply transforms if applicable
        if self.transforms is not None:
            return self.transforms(img_z, img_x, target)
        else:
            return img_z, img_x, target
    
    def __len__(self):
        return len(self.imgs)
    
    def _random_next(self):
        index = np.random.choice(len(self))
        return self.__getitem__(index)
    
    def _build_group_flags(self, imgs):
        flags = np.zeros(len(imgs), dtype=np.uint8)
        for i, name in enumerate(imgs.img_names):
            meta = imgs.img_dict[name]['target']['meta']
            if meta['width'] / meta['height'] > 1:
                flags[i] = 1
        assert len(flags) == len(self)
        return flags


@registry.register_module
class Seq2Instance(InstanceDataset):

    def __init__(self, seqs, transforms=None, sampling_stride=1):
        assert sampling_stride > 0
        super(Seq2Instance, self).__init__(
            name='{}_instances'.format(seqs.name),
            seqs=seqs,
            sampling_stride=sampling_stride)
        self.seqs = seqs
        self.transforms = transforms
        self.sampling_stride = sampling_stride
        
        # group sequences by aspect ratios (for sampling)
        self.group_flags = self._build_group_flags(
            self.ins_names, self.ins_dict)
    
    def _construct_ins_dict(self, seqs, sampling_stride):
        # construct ins_dict
        ins_dict = OrderedDict()
        for s, (img_files, target) in enumerate(seqs):
            seq_name = seqs.seq_names[s]
            if s % 100 == 0 or (s + 1) == len(seqs):
                ops.sys_print('Processing [%d/%d]: %s...' % (
                    s + 1, len(seqs), seq_name))
            
            # filter out invalid frames
            anno, meta = target['anno'], target['meta']
            mask = self._filter(anno, meta)
            anno = anno[mask]

            for f, img_file in enumerate(img_files):
                if f % sampling_stride != 0:
                    continue
                bbox = target['anno'][f]
                ins_id, cam_id = s + 1, 1
                meta_info = {
                    'width': meta['width'],
                    'height': meta['height']}

                # updat ins_dict
                name = '{}-{}_{}'.format(ins_id, cam_id, f + 1)
                ins_dict[name] = {
                    'img_file': img_file,
                    'target': {
                        'bbox': bbox,
                        'ins_id': ins_id,
                        'cam_id': cam_id,
                        'frame_id': f + 1,
                        'meta': meta_info}}
        
        return ins_dict
    
    def _filter(self, anno, meta):
        img_size = np.array(
            [[meta['width'], meta['height']]], dtype=np.float32)
        sizes = anno[:, 2:] - anno[:, :2] + 1
        areas = sizes.prod(axis=1)

        # conditions
        conditions = [
            areas > 20,
            np.all(sizes >= 10, axis=1),
            np.all(sizes <= 960, axis=1),
            np.all(sizes / img_size >= 0.01, axis=1),
            np.all(sizes / img_size <= 0.75, axis=1),
            sizes[:, 0] / np.maximum(1, sizes[:, 1]) >= 0.2,
            sizes[:, 0] / np.maximum(1, sizes[:, 1]) < 5]
        if 'cover' in meta:
            cover = meta['cover']
            conditions.append(
                cover > max(1., cover.max() * 0.3))
        
        return np.logical_and.reduce(conditions, axis=0)
    
    def _build_group_flags(self, ins_names, ins_dict):
        flags = np.zeros(len(ins_names), dtype=np.uint8)
        for i, name in enumerate(ins_names):
            ins_info = ins_dict[name]
            meta = ins_info['target']['meta']
            if meta['width'] / meta['height'] > 1:
                flags[i] = 1
        assert len(flags) == len(self)
        return flags


@registry.register_module
class Subset(Dataset):

    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices
        self.name = '{}_subset'.format(dataset.name)
    
    def __getitem__(self, index):
        return self.dataset[self.indices[index]]
    
    def __len__(self):
        return len(self.indices)
    
    def __getattr__(self, name):
        try:
            return object.__getattribute__(self, name)
        except AttributeError:
            return object.__getattribute__(self.dataset, name)


@registry.register_module
class Slice(Dataset):

    def __init__(self, dataset, start=None, stop=None, step=None):
        self.dataset = dataset
        self.indices = np.arange(start, stop, step)
        self.name = '{}_slice_{}_{}_{}'.format(
            self.dataset.name,
            start or 0,
            stop or len(self.dataset),
            step or 1)
    
    def __getitem__(self, index):
        return self.dataset[self.indices[index]]
    
    def __len__(self):
        return len(self.indices)
    
    def __getattr__(self, name):
        try:
            return object.__getattribute__(self, name)
        except AttributeError:
            return object.__getattribute__(self.dataset, name)


@registry.register_module
class RandomConcat(Dataset):

    def __init__(self, datasets, sampling_prob=None, max_size=None):
        names = [u.name for u in datasets]
        self.name = 'RandomConcat_' + '_'.join(names)
        self.datasets = datasets
        if sampling_prob is None:
            sampling_prob = np.ones(len(datasets), dtype=np.float32)
            sampling_prob /= sampling_prob.sum()
        assert len(sampling_prob) == len(datasets)
        assert (sum(sampling_prob) - 1) < 1e-6
        self.sampling_prob = np.array(sampling_prob, dtype=np.float32)
        self.max_size = max_size
        self.group_flags = self._concat_group_flags(
            self.datasets, self.sampling_prob)
    
    def __getitem__(self, index):
        d_index = np.random.choice(
            len(self.datasets), p=self.sampling_prob)
        dataset = self.datasets[d_index]

        # ensure to select the item with correct flag
        flag = self.group_flags[index]
        indices = np.where(dataset.group_flags == flag)[0]
        if len(indices) == 0:
            return self._random_next()
        index = indices[index % len(indices)]

        return dataset[index]
    
    def __len__(self):
        return len(self.group_flags)
    
    def _concat_group_flags(self, datasets, sampling_prob):
        bin_counts = np.array([np.bincount(d.group_flags, minlength=2)
                               for d in datasets], dtype=np.float32)
        prob = np.sum(bin_counts[:, 1] * \
            sampling_prob / bin_counts.sum(axis=1))
        
        # expected dataset length
        size = int(sum([len(d) * p for d, p in zip(
            datasets, sampling_prob)]))
        if self.max_size is not None:
            size = max(self.max_size, int(size))

        # generate flags according to prob
        flags = np.zeros(size, dtype=np.uint8)
        indices = np.random.choice(
            size, int(size * prob), replace=False)
        flags[indices] = 1

        return flags
    
    def _random_next(self):
        index = np.random.choice(len(self))
        return self.__getitem__(index)
