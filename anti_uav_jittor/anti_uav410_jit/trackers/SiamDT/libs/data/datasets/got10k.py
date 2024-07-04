import os.path as osp
import glob
import numpy as np

import libs.ops as ops
from libs.config import registry
from .dataset import SeqDataset


__all__ = ['GOT10k']


@registry.register_module
class GOT10k(SeqDataset):
    r"""`GOT-10K <http://got-10k.aitestunion.com//>`_ Dataset.

    Publication:
        ``GOT-10k: A Large High-Diversity Benchmark for Generic Object
        Tracking in the Wild``, L. Huang, X. Zhao and K. Huang, arXiv 2018.
    
    Args:
        root_dir (string): Root directory of dataset where ``train``,
            ``val`` and ``test`` folders exist.
        subset (string, optional): Specify ``train``, ``val`` or ``test``
            subset of GOT-10k.
        list_file (string, optional): If provided, only read sequences
            specified by the file instead of all sequences in the subset.
    """
    def __init__(self, root_dir=None, subset='test', list_file=None):
        # dataset name and paths
        if root_dir is None:
            root_dir = osp.expanduser('~/data/GOT-10k')
        assert subset in ['train', 'val', 'test']
        if list_file is None:
            list_file = osp.join(root_dir, subset, 'list.txt')
        self.root_dir = root_dir
        self.subset = subset
        self.list_file = list_file
        self.name = 'GOT-10k_{}'.format(subset)

        # initialize dataset
        super(GOT10k, self).__init__(
            self.name,
            root_dir=self.root_dir,
            subset=self.subset,
            list_file=self.list_file)
    
    def _construct_seq_dict(self, root_dir, subset, list_file):
        # image and annotation paths
        with open(list_file, 'r') as f:
            seq_names = f.read().strip().split('\n')
        seq_dirs = [osp.join(root_dir, subset, s)
                    for s in seq_names]
        anno_files = [osp.join(d, 'groundtruth.txt')
                      for d in seq_dirs]
        
        # construct seq_dict
        seq_dict = {}
        for s, seq_name in enumerate(seq_names):
            if s % 100 == 0 or s + 1 == len(seq_names):
                ops.sys_print('Processing sequence [%d/%d]: %s...' % (
                    s + 1, len(seq_names), seq_name))
            img_files = sorted(glob.glob(
                osp.join(seq_dirs[s], '*.jpg')))
            anno = np.loadtxt(
                anno_files[s], delimiter=',', dtype=np.float32)
            if anno.ndim == 1:
                assert anno.size == 4
                anno = anno[np.newaxis, :]
            anno[:, 2:] = anno[:, :2] + anno[:, 2:] - 1

            # meta information
            seq_len = len(img_files)
            img0 = ops.read_image(img_files[0])
            meta = self._fetch_meta(seq_dirs[s])
            meta.update({
                'width': img0.shape[1],
                'height': img0.shape[0],
                'frame_num': seq_len,
                'target_num': 1,
                'total_instances': seq_len})
            
            # update seq_dict
            seq_dict[seq_name] = {
                'img_files': img_files,
                'target': {
                    'anno': anno,
                    'meta': meta}}
        
        return seq_dict

    def _fetch_meta(self, seq_dir):
        # meta information
        meta_file = osp.join(seq_dir, 'meta_info.ini')
        if osp.exists(meta_file):
            with open(meta_file) as f:
                meta = f.read().strip().split('\n')[1:]
            meta = [line.split(': ') for line in meta]
            meta = {line[0]: line[1] for line in meta}
        else:
            meta = {}

        # attributes
        attributes = ['cover', 'absence', 'cut_by_image']
        for att in attributes:
            att_file = osp.join(seq_dir, att + '.label')
            if osp.exists(att_file):
                meta[att] = np.loadtxt(att_file)

        return meta
