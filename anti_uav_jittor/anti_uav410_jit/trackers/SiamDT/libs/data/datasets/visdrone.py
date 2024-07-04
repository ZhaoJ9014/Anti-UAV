import os.path as osp
import glob
import numpy as np
from collections import OrderedDict

import libs.ops as ops
from libs.config import registry
from .dataset import SeqDataset


__all__ = ['VisDroneSOT', 'VisDroneVID']


@registry.register_module
class VisDroneSOT(SeqDataset):
    """`VisDrone <http://www.aiskyeye.com/>`_ Dataset.

    Publication:
        ``Vision Meets Drones: A Challenge``,
        P. Zhu, L. Wen, X. Bian, H. Ling and Q. Hu, arXiv 2018.
    
    Args:
        root_dir (string): Root directory of dataset where subset
            folders exist.
        subset (string, optional): Specify ``train``, ``val`` or ``test``
            subset of VisDrone dataset.
    """
    def __init__(self, root_dir=None, subset='val'):
        assert subset in ['train', 'val', 'test']
        if root_dir is None:
            root_dir = osp.expanduser('~/data/VisDrone')
        if subset == 'train':
            root_dir = osp.join(root_dir, 'VisDrone2018-SOT-train')
        elif subset == 'val':
            root_dir = osp.join(root_dir, 'VisDrone2018-SOT-val')
        elif subset == 'test':
            root_dir = osp.join(root_dir, 'VisDrone2019-SOT-test-challenge')
        self.root_dir = root_dir
        self.subset = subset

        # initialize the dataset
        super(VisDroneSOT, self).__init__(
            name='VisDroneSOT_{}'.format(subset),
            root_dir=self.root_dir,
            subset=self.subset)
    
    def _construct_seq_dict(self, root_dir, subset):
        # image and annotation paths
        if subset == 'test':
            anno_files = sorted(glob.glob(
                osp.join(root_dir, 'initialization/*_s.txt')))
        else:
            anno_files = sorted(glob.glob(
                osp.join(root_dir, 'annotations/*_s.txt')))
        seq_names = [osp.basename(f)[:-4] for f in anno_files]
        seq_dirs = [osp.join(
            root_dir, 'sequences/{}'.format(n)) for n in seq_names]
        
        # construct seq_dict
        seq_dict = {}
        for s, seq_name in enumerate(seq_names):
            img_files = sorted(glob.glob(
                osp.join(seq_dirs[s], 'img*.jpg')))
            anno = np.loadtxt(anno_files[s], delimiter=',')
            if anno.ndim == 1:
                assert anno.size == 4
                anno = anno[np.newaxis, :]
            anno[:, 2:] = anno[:, :2] + anno[:, 2:] - 1
            
            # meta information
            seq_len = len(img_files)
            img0 = ops.read_image(img_files[0])
            meta = {
                'width': img0.shape[1],
                'height': img0.shape[0],
                'frame_num': seq_len,
                'target_num': 1,
                'total_instances': seq_len}
            
            # update seq_dict
            seq_dict[seq_name] = {
                'img_files': img_files,
                'target': {
                    'anno': anno,
                    'meta': meta}}
        
        return seq_dict


@registry.register_module
class VisDroneVID(SeqDataset):
    r"""`VisDrone VID <http://www.aiskyeye.com/>`_ Dataset.

    Publication:
        ``Vision Meets Drones: A Challenge``, P. Zhu, L. Wen, X. Bian, H. Ling and Q. Hu, arXiv 2018.
    
    Args:
        root_dir (string): Root directory of dataset where ``sequences`` and
            ``annotations`` folders exist.
        subset (string, optional): Specify ``train``, ``val`` or (``train``, ``val``)
            subset(s) of VisDrone VID. Default is a tuple (``train``, ``val``).
    """
    CLASSES = (
        'ignored regions',  # 0
        'pedestrian',       # 1
        'people',           # 2
        'bicycle',          # 3
        'car',              # 4
        'van',              # 5
        'truck',            # 6
        'tricycle',         # 7
        'awning-tricycle',  # 8
        'bus',              # 9
        'motor',            # 10
        'others')           # 11
    
    def __init__(self, root_dir=None, subset=['train', 'val']):
        # dataset name and paths
        if root_dir is None:
            root_dir = osp.expanduser('~/data/VisDrone')
        if isinstance(subset, str):
            assert subset in ['train', 'val']
            subset = [subset]
        elif isinstance(subset, (list, tuple)):
            assert all([s in ['train', 'val'] for s in subset])
            subset = subset
        else:
            raise Exception('Unknown subset')
        self.root_dir = root_dir
        self.subset = subset
        self.name = 'VisDroneVID_{}'.format('_'.join(subset))

        # initialize dataset
        super(VisDroneVID, self).__init__(
            self.name,
            root_dir=self.root_dir,
            subset=self.subset)
    
    def _construct_seq_dict(self, root_dir, subset):
        # image and annotation paths
        seq_dirs = []
        anno_files = []
        if 'train' in subset:
            _seq_dirs = sorted(glob.glob(osp.join(
                root_dir, 'VisDrone2018-VID-train/sequences/*_v')))
            _anno_files = [osp.join(
                root_dir, 'VisDrone2018-VID-train/annotations',
                osp.basename(s) + '.txt') for s in _seq_dirs]
            seq_dirs += _seq_dirs
            anno_files += _anno_files
        if 'val' in subset:
            _seq_dirs = sorted(glob.glob(osp.join(
                root_dir, 'VisDrone2018-VID-val/sequences/*_v')))
            _anno_files = [osp.join(
                root_dir, 'VisDrone2018-VID-val/annotations',
                osp.basename(s) + '.txt') for s in _seq_dirs]
            seq_dirs += _seq_dirs
            anno_files += _anno_files
        seq_names = [osp.basename(s) for s in seq_dirs]

        # construct seq_dict
        seq_dict = {}
        for s, seq_name in enumerate(seq_names):
            if s % 10 == 0 or s + 1 == len(seq_names):
                ops.sys_print('Processing [%d/%d]: %s' % (
                    s + 1, len(seq_names), seq_name))
            
            img_files = sorted(glob.glob(
                osp.join(seq_dirs[s], '*.jpg')))
            anno_s = np.loadtxt(
                anno_files[s], delimiter=',', dtype=np.float32)
            anno_s = self._format(anno_s)

            # meta information
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
    
    def _format(self, anno):
        r"""Standadize the gt format.

        Input format:
            frame_id (1-indexed), target_id, x1, y1, w, h, score/keep,
                class_id, truncation, occlusion
        
        Output format:
            frame_id (0-indexed), target_id, x1, y1, x2, y2, keep,
                class_id, occlusion
        """
        col_indices = [0, 1, 2, 3, 4, 5, 6, 7, 9]
        anno = anno[:, col_indices]

        anno[:, 0] -= 1     # convert frame_id to 0-indexed
        anno[:, 4:6] = anno[:, 2:4] + anno[:, 4:6] - 1
        anno[:, 8] /= 2.    # normalize occlusion to [0, 1]
        
        # only keep meaningful annotations
        # (filter by classes and occlusions)
        mask = (anno[:, 7] > 0) & (anno[:, 7] < 11) & (anno[:, 8] < 1)
        anno = anno[mask, :]

        return anno
