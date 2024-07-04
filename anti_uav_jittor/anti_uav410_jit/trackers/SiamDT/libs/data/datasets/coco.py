import os.path as osp
import numpy as np
from pycocotools.coco import COCO

import libs.ops as ops
from libs.config import registry
from .dataset import ImageDataset


__all__ = ['COCODetection']


@registry.register_module
class COCODetection(ImageDataset):
    r"""`Common Objects in Context (COCO) <http://cocodataset.org/>`_ Dataset.

    Publication:
        ``Microsoft COCO: Common Objects in Context``, T. Y. Lin, M. Maire, S. Belongie, et. al., arXiv 2014.
    
    Args:
        root_dir (string): Root directory of dataset where ``Data`` and
            ``Annotations`` folders exist.
        version (integer, optional): Specify the dataset version. Specify as
            one of 2014, 2015 or 2017. Default is 2017.
        subset (string, optional): Specify ``train`` or ``val`` subset of
            COCO. Default is ``val``.
        transforms (object, optional): Augmentations applied to each dataset item.
            Default is None.
    """
    CLASSES = ('person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
               'train', 'truck', 'boat', 'traffic_light', 'fire_hydrant',
               'stop_sign', 'parking_meter', 'bench', 'bird', 'cat', 'dog',
               'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
               'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
               'skis', 'snowboard', 'sports_ball', 'kite', 'baseball_bat',
               'baseball_glove', 'skateboard', 'surfboard', 'tennis_racket',
               'bottle', 'wine_glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
               'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
               'hot_dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
               'potted_plant', 'bed', 'dining_table', 'toilet', 'tv', 'laptop',
               'mouse', 'remote', 'keyboard', 'cell_phone', 'microwave',
               'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
               'vase', 'scissors', 'teddy_bear', 'hair_drier', 'toothbrush')

    def __init__(self, root_dir=None, version=2017, subset='val',
                 transforms=None):
        if root_dir is None:
            root_dir = osp.expanduser('~/data/coco')
        super(COCODetection, self).__init__(
            name='COCO{}_{}'.format(version, subset),
            root_dir=root_dir,
            version=version,
            subset=subset)
        self.transforms = transforms
    
    def _construct_img_dict(self, root_dir, version, subset):
        # image and annotation paths
        img_dir = osp.join(
            root_dir,
            '{}{}'.format(subset, version))
        ann_file = osp.join(
            root_dir,
            'annotations/instances_{}{}.json'.format(subset, version))
        coco = COCO(ann_file)
        img_infos = coco.dataset['images']
        # filter small images
        img_infos = [u for u in img_infos
                     if min(u['width'], u['height']) >= 32]
        img_ids = [u['id'] for u in img_infos]

        # make class IDs contiguous
        self._cat2id = {
            v: i + 1 for i, v in enumerate(coco.getCatIds())}
        self._id2cat = {v: k for k, v in self._cat2id.items()}

        # construct img_dict
        img_dict = {}
        for i, img_id in enumerate(img_ids):
            if i % 1000 == 0 or i + 1 == len(img_ids):
                ops.sys_print('Processing image [%d/%d]: %d...' % (
                    i + 1, len(img_ids), img_id))
            
            # load annotation
            ann_id = coco.getAnnIds(imgIds=img_id)
            anno = coco.loadAnns(ann_id)
            anno = [obj for obj in anno if self._check_obj(obj)]
            if len(anno) == 0:
                continue

            # read image
            img_file = coco.loadImgs(img_id)[0]['file_name']
            img_file = osp.join(img_dir, img_file)
            
            # read bboxes, labels and mask polygons
            bboxes = [obj['bbox'] for obj in anno]
            bboxes = np.array(bboxes, dtype=np.float32).reshape(-1, 4)
            bboxes[:, 2:] = bboxes[:, :2] + bboxes[:, 2:] - 1

            labels = [obj['category_id'] for obj in anno]
            labels = [self._cat2id[c] for c in labels]
            labels = np.array(labels, dtype=np.int64)

            mask_polys = [obj['segmentation'] for obj in anno]
            for j, poly in enumerate(mask_polys):
                # valid polygons have >= 3 points (6 coordinates)
                mask_polys[j] = [np.array(p) for p in poly if len(p) > 6]
            
            # update img_dict
            img_dict[img_id] = {
                'img_file': img_file,
                'target': {
                    'bboxes': bboxes,
                    'labels': labels,
                    'mask_polys': mask_polys,
                    'meta': img_infos[i]}}
    
        return img_dict

    def _check_obj(self, obj):
        _, _, w, h = obj['bbox']
        ignore = obj.get('ignore', False)
        if obj['iscrowd'] or ignore or w < 1 or h < 1:
            return False
        return True
