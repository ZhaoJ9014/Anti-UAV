import os.path as osp
import glob
import numpy as np

import libs.ops as ops
from libs.config import registry
from .dataset import SeqDataset


__all__ = ['NfS']


@registry.register_module
class NfS(SeqDataset):
    """`NfS <http://ci2cv.net/nfs/index.html>`_ Dataset.

    Publication:
        ``Need for Speed: A Benchmark for Higher Frame Rate Object Tracking``,
        H. K. Galoogahi, A. Fagg, C. Huang, D. Ramanan and S. Lucey, ICCV 2017.
    
    Args:
        root_dir (string): Root directory of dataset where sequence
            folders exist.
        fps (integer): Sequence frame rate. Two options ``30`` and ``240``
            are available. Default is 240.
    """
    def __init__(self, root_dir=None, fps=30):
        assert fps in [30, 240]
        if root_dir is None:
            root_dir = osp.expanduser('~/data/nfs')
        self.root_dir = root_dir
        self.fps = fps

        # initialize the dataset
        super(NfS, self).__init__(
            name='NfS_{}'.format(fps),
            root_dir=root_dir,
            fps=fps)

    def _construct_seq_dict(self, root_dir, fps):
        # image and annotation paths
        anno_files = sorted(glob.glob(
            osp.join(root_dir, '*/%d/*.txt' % fps)))
        seq_names = [
            osp.basename(f)[:-4] for f in anno_files]
        seq_dirs = [osp.join(
            osp.dirname(f), n)
            for f, n in zip(anno_files, seq_names)]
        
        # construct seq_dict
        seq_dict = {}
        for s, seq_name in enumerate(seq_names):
            if s % 50 == 0 or s + 1 == len(seq_names):
                ops.sys_print('Processing sequence [%d/%d]: %s...' % (
                    s + 1, len(seq_names), seq_name))
            img_files = sorted(glob.glob(
                osp.join(seq_dirs[s], '*.jpg')))
            anno = np.loadtxt(anno_files[s], dtype=str)
            anno = anno[:, 1:5].astype(np.float32)

            # handle inconsistent lengths
            if not len(img_files) == len(anno):
                if abs(len(anno) / len(img_files) - 8) < 1:
                    anno = anno[0::8, :]
                diff = abs(len(img_files) - len(anno))
                if diff > 0 and diff <= 1:
                    n = min(len(img_files), len(anno))
                    anno = anno[:n]
                    img_files = img_files[:n]
            assert len(img_files) == len(anno)

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
