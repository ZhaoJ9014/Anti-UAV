import os.path as osp
import glob
import json
import numpy as np

import libs.ops as ops
from libs.config import registry
from .dataset import SeqDataset


__all__ = ['LaSOT']


@registry.register_module
class LaSOT(SeqDataset):
    r"""`LaSOT <https://cis.temple.edu/lasot/>`_ Datasets.

    Publication:
        ``LaSOT: A High-quality Benchmark for Large-scale Single Object Tracking``,
        H. Fan, L. Lin, F. Yang, P. Chu, G. Deng, S. Yu, H. Bai,
        Y. Xu, C. Liao, and H. Ling., CVPR 2019.
    
    Args:
        root_dir (string): Root directory of dataset where sequence
            folders exist.
        subset (string, optional): Specify ``train`` or ``test``
            subset of LaSOT.
    """
    def __init__(self, root_dir=None, subset='test'):
        # dataset name and paths
        assert subset in ['train', 'test'], 'Unknown subset.'
        if root_dir is None:
            root_dir = osp.expanduser('~/data/LaSOTBenchmark')
        self.root_dir = root_dir
        self.subset = subset
        self.name = 'LaSOT_{}'.format(subset)

        # initialize dataset
        super(LaSOT, self).__init__(
            self.name,
            root_dir=self.root_dir,
            subset=self.subset)
    
    def _construct_seq_dict(self, root_dir, subset):
        # load subset sequence names
        split_file = osp.join(
            osp.dirname(__file__), 'lasot.json')
        with open(split_file, 'r') as f:
            splits = json.load(f)
        seq_names = splits[subset]

        # image and annotation paths
        seq_dirs = [osp.join(
            root_dir, n[:n.rfind('-')], n, 'img')
            for n in seq_names]
        anno_files = [osp.join(
            root_dir, n[:n.rfind('-')], n, 'groundtruth.txt')
            for n in seq_names]
        
        # construct seq_dict
        seq_dict = {}
        for s, seq_name in enumerate(seq_names):
            if s % 100 == 0 or s + 1 == len(seq_names):
                ops.sys_print('Processing sequence [%d/%d]: %s...' % (
                    s + 1, len(seq_names), seq_name))
            img_files = sorted(glob.glob(osp.join(
                seq_dirs[s], '*.jpg')))
            anno = np.loadtxt(
                anno_files[s], delimiter=',', dtype=np.float32)
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
        seq_dir = osp.dirname(seq_dir)
        meta = {}

        # attributes
        for att in ['full_occlusion', 'out_of_view']:
            att_file = osp.join(seq_dir, att + '.txt')
            if osp.exists(att_file):
                meta[att] = np.loadtxt(att_file, delimiter=',')
        
        # nlp
        nlp_file = osp.join(seq_dir, 'nlp.txt')
        with open(nlp_file, 'r') as f:
            if osp.exists(nlp_file):
                meta['nlp'] = f.read().strip()

        return meta
