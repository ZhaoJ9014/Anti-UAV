import os
import os.path as osp
import glob
import numpy as np

import libs.ops as ops
from libs.config import registry
from .dataset import SeqDataset


__all__ = ['TColor128']


@registry.register_module
class TColor128(SeqDataset):
    """`TColor128 <http://www.dabi.temple.edu/~hbling/data/TColor-128/TColor-128.html>`_ Dataset.

    Publication:
        ``Encoding color information for visual tracking: algorithms and benchmark``,
        P. Liang, E. Blasch and H. Ling, TIP, 2015.
    
    Args:
        root_dir (string): Root directory of dataset where sequence
            folders exist.
    """
    def __init__(self, root_dir=None, download=True):
        if root_dir is None:
            root_dir = osp.expanduser('~/data/Temple-color-128')
        self.root_dir = root_dir
        if download:
            self._download(root_dir)
        
        # initialize the dataset
        super(TColor128, self).__init__(
            name='TColor-128',
            root_dir=root_dir)

    def _construct_seq_dict(self, root_dir):
        # image and annotation paths
        anno_files = sorted(glob.glob(
            osp.join(root_dir, '*/*_gt.txt')))
        seq_dirs = [osp.dirname(f) for f in anno_files]
        seq_names = [osp.basename(d) for d in seq_dirs]
        # valid frame range for each sequence
        range_files = [glob.glob(
            osp.join(d, '*_frames.txt'))[0]
            for d in seq_dirs]
        
        # construct seq_dict
        seq_dict = {}
        for s, seq_name in enumerate(seq_names):
            # load valid frame range
            frames = np.loadtxt(
                range_files[s], dtype=int, delimiter=',')
            img_files = [osp.join(
                seq_dirs[s], 'img/%04d.jpg' % f)
                for f in range(frames[0], frames[1] + 1)]

            # load annotations
            anno = np.loadtxt(anno_files[s], delimiter=',')
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

    def _download(self, root_dir):
        if not osp.isdir(root_dir):
            os.makedirs(root_dir)
        elif len(os.listdir(root_dir)) > 100:
            ops.sys_print('Files already downloaded.')
            return

        url = 'http://www.dabi.temple.edu/~hbling/data/TColor-128/Temple-color-128.zip'
        zip_file = osp.join(root_dir, 'Temple-color-128.zip')
        ops.sys_print('Downloading to %s...' % zip_file)
        ops.download(url, zip_file)
        ops.sys_print('\nExtracting to %s...' % root_dir)
        ops.extract(zip_file, root_dir)

        return root_dir
