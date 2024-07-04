import os.path as osp
import glob
import numpy as np

import libs.ops as ops
from libs.config import registry
from .dataset import SeqDataset


__all__ = ['POT']


@registry.register_module
class POT(SeqDataset):
    """`POT <http://www.dabi.temple.edu/~hbling/data/POT-210/planar_benchmark.html>`_ Dataset.

    Publication:
        ``Planar Object Tracking in the Wild: A Benchmark``,
        P. Liang, Y. Wu, H. Lu, L. Wang, C. Liao, and H. Ling, ICRA 2018.
    
    Args:
        root_dir (string): Root directory of dataset where sequence
            folders exist.
    """
    def __init__(self, root_dir=None):
        if root_dir is None:
            root_dir = osp.expanduser('~/data/POT')
        self.root_dir = root_dir

        # initialize the dataset
        super(POT, self).__init__(
            name='POT',
            root_dir=root_dir)

    def _construct_seq_dict(self, root_dir):
        # image and annotation paths
        seq_dirs = sorted(glob.glob(
            osp.join(root_dir, '*/*_*/')))
        seq_dirs = [d[:-1] for d in seq_dirs]
        seq_names = [osp.basename(d) for d in seq_dirs]
        anno_files = [osp.join(
            root_dir, 'annotation/annotation/{}_gt_points.txt'.format(n))
            for n in seq_names]
        
        # construct seq_dict
        seq_dict = {}
        for s, seq_name in enumerate(seq_names):
            if s % 50 == 0 or s + 1 == len(seq_names):
                ops.sys_print('Processing sequence [%d/%d]: %s...' % (
                    s + 1, len(seq_names), seq_name))
            
            img_files = sorted(glob.glob(
                osp.join(seq_dirs[s], '*.jpg')))
            anno = np.loadtxt(anno_files[s])
            
            n = min(len(img_files), len(anno))
            assert n > 0
            img_files = img_files[:n]
            anno = anno[:n]

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
