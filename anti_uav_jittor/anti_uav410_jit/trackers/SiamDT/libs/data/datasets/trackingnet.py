import os.path as osp
import glob
import numpy as np

import libs.ops as ops
from libs.config import registry
from .dataset import SeqDataset


__all__ = ['TrackingNet']


@registry.register_module
class TrackingNet(SeqDataset):
    r"""`TrackingNet <https://tracking-net.org/>`_ Datasets.

    Publication:
        ``TrackingNet: A Large-Scale Dataset and Benchmark for Object Tracking in the Wild.``,
        M. Muller, A. Bibi, S. Giancola, S. Al-Subaihi and B. Ghanem, ECCV 2018.
    
    Args:
        root_dir (string): Root directory of dataset where sequence
            folders exist.
        subset (string, optional): Specify ``train`` or ``test``
            subset of TrackingNet.
    """
    def __init__(self, root_dir=None, subset='test'):
        assert subset in ['train', 'test'], 'Unknown subset.'
        if root_dir is None:
            root_dir = osp.expanduser('~/data/TrackingNet')
        self.root_dir = root_dir
        self.subset = subset
        if subset == 'test':
            subset_dirs = ['TEST']
        elif subset == 'train':
            subset_dirs = ['TRAIN_%d' % c for c in range(12)]
        
        # initialize the dataset
        super(TrackingNet, self).__init__(
            name='TrackingNet_{}'.format(self.subset),
            root_dir=self.root_dir,
            subset_dirs=subset_dirs)

    def _construct_seq_dict(self, root_dir, subset_dirs):
        # image and annotation paths
        anno_files = [glob.glob(osp.join(
            root_dir, c, 'anno/*.txt')) for c in subset_dirs]
        anno_files = sorted(sum(anno_files, []))
        seq_dirs = [osp.join(
            osp.dirname(osp.dirname(f)),
            'frames', osp.basename(f)[:-4])
            for f in anno_files]
        seq_names = [osp.basename(d) for d in seq_dirs]

        # construct seq_dict
        seq_dict = {}
        for s, seq_name in enumerate(seq_names):
            if s % 100 == 0 or s + 1 == len(seq_names):
                ops.sys_print('Processing sequence [%d/%d]: %s...' % (
                    s + 1, len(seq_names), seq_name))
            img_files = glob.glob(
                osp.join(seq_dirs[s], '*.jpg'))
            img_files = sorted(
                img_files,
                key=lambda f: int(osp.basename(f)[:-4]))
            anno = np.loadtxt(anno_files[s], delimiter=',')
            if anno.ndim == 1:
                anno = np.expand_dims(anno, axis=0)
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
