import os.path as osp
import glob
import numpy as np
import xml.etree.ElementTree as ET

import libs.ops as ops
from libs.config import registry
from .dataset import SeqDataset


__all__ = ['ImageNetVID']


@registry.register_module
class ImageNetVID(SeqDataset):
    r"""`ImageNet Video Image Detection (VID) <https://image-net.org/challenges/LSVRC/2015/#vid>`_ Dataset.

    Publication:
        ``ImageNet Large Scale Visual Recognition Challenge``, O. Russakovsky,
            J. deng, H. Su, etc. IJCV, 2015.
    
    Args:
        root_dir (string): Root directory of dataset where ``Data`` and
            ``Annotations`` folders exist.
        subset (string, optional): Specify ``train``, ``val`` or (``train``, ``val``)
            subset(s) of ImageNet-VID. Default is a tuple (``train``, ``val``).
    """
    CLASSES = (
        'airplane',
        'antelope',
        'bear ',
        'bicycle',
        'bird',
        'bus',
        'car',
        'cattle',
        'dog',
        'domestic cat',
        'elephant',
        'fox',
        'giant panda',
        'hamster',
        'horse',
        'lion',
        'lizard',
        'monkey',
        'motorcycle ',
        'rabbit',
        'red panda',
        'sheep',
        'snake',
        'squirrel',
        'tiger',
        'train',
        'turtle',
        'watercraft ',
        'whale',
        'zebra')
    WORDNET_IDs = (
        'n02691156',
        'n02419796',
        'n02131653',
        'n02834778',
        'n01503061',
        'n02924116',
        'n02958343',
        'n02402425',
        'n02084071',
        'n02121808',
        'n02503517',
        'n02118333',
        'n02510455',
        'n02342885',
        'n02374451',
        'n02129165',
        'n01674464',
        'n02484322',
        'n03790512',
        'n02324045',
        'n02509815',
        'n02411705',
        'n01726692',
        'n02355227',
        'n02129604',
        'n04468005',
        'n01662784',
        'n04530566',
        'n02062744',
        'n02391049')

    def __init__(self, root_dir=None, subset=['train', 'val']):
        # dataset name and paths
        if root_dir is None:
            root_dir = osp.expanduser('~/data/ILSVRC')
        if isinstance(subset, str):
            assert subset in ['train', 'val']
            subset = [subset]
        elif isinstance(subset, (list, tuple)):
            assert all([s in ['train', 'val'] for s in subset])
            subset = list(subset)
        else:
            raise ValueError('Unknown subset')
        self.root_dir = root_dir
        self.subset = subset
        self.name = 'ImageNetVID_{}'.format('_'.join(subset))

        # initialize dataset
        super(ImageNetVID, self).__init__(
            self.name,
            root_dir=self.root_dir,
            subset=self.subset)

    def _construct_seq_dict(self, root_dir, subset):
        # image and annotation paths
        seq_dirs = []
        anno_dirs = []
        if 'train' in subset:
            _seq_dirs = sorted(glob.glob(osp.join(
                root_dir, 'Data/VID/train/ILSVRC*/ILSVRC*')))
            _anno_dirs = [osp.join(
                root_dir, 'Annotations/VID/train',
                *s.split('/')[-2:]) for s in _seq_dirs]
            seq_dirs += _seq_dirs
            anno_dirs += _anno_dirs
        if 'val' in subset:
            _seq_dirs = sorted(glob.glob(osp.join(
                root_dir, 'Data/VID/val/ILSVRC2015_val_*')))
            _anno_dirs = [osp.join(
                root_dir, 'Annotations/VID/val',
                s.split('/')[-1]) for s in _seq_dirs]
            seq_dirs += _seq_dirs
            anno_dirs += _anno_dirs
        seq_names = [osp.basename(s) for s in seq_dirs]

        # construct seq_dict
        seq_dict = {}
        for s, seq_name in enumerate(seq_names):
            if s % 100 == 0 or s + 1 == len(seq_names):
                ops.sys_print('Processing sequence [%d/%d]: %s...' % (
                    s + 1, len(seq_names), seq_name))
            
            # parse XML annotation files
            anno_files = sorted(glob.glob(osp.join(
                anno_dirs[s], '*.xml')))
            objects = [ET.ElementTree(file=f).findall('object')
                       for f in anno_files]
            
            anno_s = []
            for f, group in enumerate(objects):
                for obj in group:
                    anno_obj = [
                        f,                                  # frame_id
                        int(obj.find('trackid').text),      # target_id
                        int(obj.find('bndbox/xmin').text),  # x1
                        int(obj.find('bndbox/ymin').text),  # y1
                        int(obj.find('bndbox/xmax').text),  # x2
                        int(obj.find('bndbox/ymax').text),  # y2
                        1,                                  # score/keep
                        self.WORDNET_IDs.index(obj.find('name').text),  # class_id
                        int(obj.find('occluded').text)]     # occlusion
                    anno_s.append(anno_obj)
            anno_s = np.array(anno_s, dtype=np.float32)

            # meta information
            img_files = [osp.join(seq_dirs[s], '%06d.JPEG' % f)
                         for f in range(len(objects))]
            img0 = ops.read_image(img_files[0])
            meta = {
                'width': img0.shape[1],
                'height': img0.shape[0],
                'frame_num': len(img_files),
                'target_num': len(set(anno_s[:, 1])),
                'total_instances': len(anno_s)}
            
            # update seq_dict
            seq_dict[seq_name] = {
                'img_files': img_files,
                'target': {
                    'anno': anno_s,
                    'meta': meta}}

        return seq_dict
