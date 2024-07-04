import os.path as osp
import glob
import numpy as np

import libs.ops as ops
from libs.config import registry
from .dataset import SeqDataset


@registry.register_module
class MOT(SeqDataset):
    """`MOT <https://motchallenge.net/>`_ Dataset.

    Publication:
        ``MOT16: A Benchmark for Multi-Object Tracking``,
        Milan, A., Leal-Taixé, L., Reid, I., Roth, S. and Schindler, K., arXiv 2016.
    
    Args:
        root_dir (string): Root directory of dataset where sequence
            folders exist.
        version (integer, optional): Specify the version of MOT. Specify as
            one of 2015, 2016, 2017 and 2019.
        subset (string, optional): Specify ``train`` or ``test``
            subset of MOT.
    """
    CLASSES = (
        'Pedestrian',           # 0
        'Person on vehicle',    # 1
        'Car',                  # 2
        'Bicycle',              # 3
        'Motorbike',                # 4
        'Non motorized vehicle',    # 5
        'Static person',        # 6
        'Distractor',           # 7
        'Occluder',             # 8
        'Occluder on the ground',   # 9
        'Occluder full',        # 10
        'Reflection',           # 11
        'Crowd')                # 12

    def __init__(self, root_dir=None, version=2016, subset='train'):
        assert version in [2015, 2016, 2017, 2019]
        assert subset in ['train', 'test']
        if version == 2015:
            name = '2DMOT2015'
        else:
            name = 'MOT{}'.format(version % 100)
        if root_dir is None:
            root_dir = osp.expanduser('~/data/' + name)
        self.root_dir = root_dir
        self.version = version
        self.subset = subset
        self.name = '{}_{}'.format(name, subset)

        # initialize dataset
        super(MOT, self).__init__(
            self.name,
            root_dir=self.root_dir,
            version=self.version,
            subset=self.subset)
    
    def _construct_seq_dict(self, root_dir, version, subset):
        # image and annotation paths
        seq_dirs = sorted(glob.glob(
            osp.join(root_dir, subset, '*/img1')))
        parent_dirs = [osp.dirname(d) for d in seq_dirs]
        seq_names = [osp.basename(d) for d in parent_dirs]
        det_files = [osp.join(
            d, 'det/det.txt') for d in parent_dirs]
        if subset == 'train':
            gt_files = [osp.join(
                d, 'gt/gt.txt') for d in parent_dirs]
        
        # construct seq_dict
        seq_dict = {}
        for s, seq_name in enumerate(seq_names):
            img_files = sorted(glob.glob(
                osp.join(seq_dirs[s], '*.jpg')))
            
            det = np.loadtxt(
                det_files[s], delimiter=',', dtype=np.float32)
            det = self._format_det(det)
            target = {'det': det}

            if subset == 'train':
                gt = np.loadtxt(
                    gt_files[s], delimiter=',', dtype=np.float32)
                gt = self._format_gt(gt)
                target.update({'anno': gt})
            
            # meta information
            img0 = ops.read_image(img_files[0])
            meta = {
                'width': img0.shape[1],
                'height': img0.shape[0],
                'frame_num': len(img_files),
                'target_num': -1 if subset != 'train' else len(set(gt[:, 1])),
                'total_instances': -1 if subset != 'train' else len(gt)}
            target.update({'meta': meta})

            # update seq_dict
            seq_dict[seq_name] = {
                'img_files': img_files,
                'target': target}
        
        return seq_dict
    
    def _format_gt(self, gt):
        r"""Standadize the gt format.

        Input format:
            frame_id (1-indexed), target_id (1-indexed), x1, y1, w, h, keep,
                class_id (1-indexed), visibility ratio
        
        Output format:
            frame_id (0-indexed), target_id (0-indexed), x1, y1, x2, y2, keep,
                class_id (0-indexed), occlusion
        """
        anno = [
            gt[:, 0:1] - 1,     # frame_id
            gt[:, 1:2] - 1,     # target_id
            gt[:, 2:4],         # x1, y1
            gt[:, 2:4] + gt[:, 4:6] - 1,    # x2, y2
            gt[:, 6:7],         # keep
            gt[:, 7:8] - 1,     # class_id
            1 - gt[:, 8:9]]     # occlusion
        anno = np.concatenate(anno, axis=1)
        
        # filter invalid annotations
        mask_class = np.logical_or.reduce((
            anno[:, 7] == 0,
            anno[:, 7] == 2,
            anno[:, 7] == 3,
            anno[:, 7] == 4))
        mask_frame = (anno[:, 0] >= 0) & (anno[:, 0] < 9999)
        mask_keep = anno[:, 6] > 0.5
        mask_occ = anno[:, 8] > 0.5
        mask = (mask_class & mask_frame & mask_keep)
        anno = anno[mask]

        return anno
    
    def _format_det(self, det):
        r"""Standadize the gt format.

        Input format:
            frame_id (1-indexed), target_id (1-indexed), x1, y1, w, h,
                conf, x, y, z
        
        Output format:
            frame_id (0-indexed), target_id (0-indexed), x1, y1, x2, y2, keep,
                class_id (0-indexed), occlusion
        """
        ones = np.ones((len(det), 1), dtype=det.dtype)
        anno = [
            det[:, 0:1] - 1,    # frame_id
            ones,               # target_id
            det[:, 2:4],        # x1, y1
            det[:, 2:4] + det[:, 4:6] - 1,  # x2, y2
            ones,               # keep
            ones * 0,           # class_id
            ones * 0]           # occlusion
        anno = np.concatenate(anno, axis=1)
        
        # filter invalid annotations
        mask = (anno[:, 0] >= 0) & (anno[:, 0] < 9999)
        anno = anno[mask]

        return anno
