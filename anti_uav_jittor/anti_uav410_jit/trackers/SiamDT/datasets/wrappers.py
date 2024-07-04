import numpy as np

import libs.data as data
import libs.ops as ops
from mmcv.parallel import DataContainer as DC
from mmdet.datasets import DATASETS

__all__ = ['PairWrapper']


def _datasets(name):
    assert isinstance(name, str)
    if name == 'coco_train':
        return data.COCODetection(subset='train')
    elif name == 'coco_val':
        return data.COCODetection(subset='val')
    elif name == 'got10k_train':
        return data.GOT10k(root_dir='/media/data2/TrackingDatasets/GOT-10k/full_data/train_data', subset='train')
    elif name == 'got10k_val':
        return data.GOT10k(root_dir='/media/data2/TrackingDatasets/GOT-10k/full_data/train_data', subset='val')
    elif name == 'lasot_train':
        return data.LaSOT(root_dir='/media/data2/TrackingDatasets/LaSOT/', subset='train')
    elif name == 'uavtir_train':
        return data.UAVtir(root_dir='/media/data2/TrackingDatasets/Anti-UAV410/Anti-UAV/', subset='train')
    elif name == 'uavtir_val':
        return data.UAVtir(root_dir='/media/data2/TrackingDatasets/Anti-UAV410/Anti-UAV/', subset='val')
    elif name == 'imagenet_vid':
        return data.ImageNetVID(subset=['train', 'val'])
    elif name == 'visdrone_vid':
        return data.VisDroneVID(subset=['train', 'val'])
    else:
        raise KeyError('Unknown dataset:', name)


def _transforms(name):
    # standard size: (1333, 800)
    if name == 'basic_train':
        return data.BasicPairTransforms(train=True)
    elif name == 'basic_test':
        return data.BasicPairTransforms(train=False)
    elif name == 'extra_partial':
        return data.ExtraPairTransforms(
            with_photometric=True,
            with_expand=False,
            with_crop=False)
    elif name == 'extra_full':
        return data.ExtraPairTransforms()
    else:
        raise KeyError('Unknown transform:', name)


@DATASETS.register_module()
class PairWrapper(data.PairDataset):

    def __init__(self,
                 base_dataset='coco_train,got10k_train,lasot_train',
                 base_transforms='extra_partial',
                 sampling_prob=[0.4, 0.4, 0.2],
                 max_size=30000,
                 max_instances=8,
                 with_label=True,
                 **kwargs):
        # setup base dataset and indices (bounded by max_size)
        self.base_dataset = self._setup_base_dataset(
            base_dataset, base_transforms, sampling_prob, max_size)
        self.indices = self._setup_indices(
            self.base_dataset, max_size)
        # member variables
        self.max_size = max_size
        self.max_instances = max_instances
        self.with_label = with_label
        self.flag = self.base_dataset.group_flags[self.indices]

    def __getitem__(self, index):
        if index == 0:
            self.indices = self._setup_indices(
                self.base_dataset, self.max_size)
        index = self.indices[index]
        item = self.base_dataset[index]

        # sanity check
        keys = [
            'img_z',
            'img_x',
            'gt_bboxes_z',
            'gt_bboxes_x',
            'img_meta_z',
            'img_meta_x']
        assert [k in item for k in keys]
        assert len(item['gt_bboxes_z']) == len(item['gt_bboxes_x'])
        if len(item['gt_bboxes_z']) == 0 or \
                len(item['gt_bboxes_x']) == 0:
            return self._random_next()

        # sample up to "max_instances" instances
        if self.max_instances > 0 and \
                len(item['gt_bboxes_z']) > self.max_instances:
            indices = np.random.choice(
                len(item['gt_bboxes_z']),
                self.max_instances,
                replace=False)
            item['gt_bboxes_z'] = item['gt_bboxes_z'][indices]
            item['gt_bboxes_x'] = item['gt_bboxes_x'][indices]

        # construct DataContainer
        item = {
            'img_z': DC(item['img_z'], stack=True),
            'img_x': DC(item['img_x'], stack=True),
            'img_meta_z': DC(item['img_meta_z'], cpu_only=True),
            'img_meta_x': DC(item['img_meta_x'], cpu_only=True),
            'gt_bboxes_z': DC(item['gt_bboxes_z'].float()),
            'gt_bboxes_x': DC(item['gt_bboxes_x'].float())}

        # attach class labels if required to
        if self.with_label:
            _tmp = item['gt_bboxes_x'].data
            item['gt_labels'] = DC(_tmp.new_zeros(len(_tmp)).long())  # 原来是_tmp.new_ones

        return item

    def __len__(self):
        return len(self.indices)

    def _random_next(self):
        index = np.random.choice(len(self))
        return self.__getitem__(index)

    def _setup_indices(self, base_dataset, max_size):
        if max_size > 0 and len(base_dataset) > max_size:
            indices = np.random.choice(
                len(base_dataset), max_size, replace=False)
        else:
            indices = np.arange(len(base_dataset))
        return indices

    def _setup_base_dataset(self, base_dataset, base_transforms,
                            sampling_prob, max_size):
        names = base_dataset.split(',')
        datasets = []
        for name in names:
            if 'coco' in name:
                # image-style dataset
                dataset = data.Image2Pair(
                    _datasets(name),
                    _transforms(base_transforms))
            else:
                # sequence-style dataset
                dataset = data.Seq2Pair(
                    _datasets(name),
                    _transforms(base_transforms))
            datasets.append(dataset)

        # concatenate datasets if necessary
        if len(datasets) == 1:
            return datasets[0]
        else:
            return data.RandomConcat(datasets, sampling_prob, max_size)
