import os.path as osp
import glob
import numpy as np

import libs.ops as ops
from libs.config import registry
from .dataset import SeqDataset


__all__ = ['TLP']


@registry.register_module
class TLP(SeqDataset):
    """`TLP <https://amoudgl.github.io/tlp/>`_ Dataset.

    Publication:
        ``Long-term Visual Object Tracking Benchmark``,
        Moudgil Abhinav and Gandhi Vineet, ACCV 2018.
    
    Args:
        root_dir (string): Root directory of dataset where sequence
            folders exist.
    """
    def __init__(self, root_dir=None):
        if root_dir is None:
            root_dir = osp.expanduser('~/data/TLP')
        self.root_dir = root_dir

        # initialize the dataset
        super(TLP, self).__init__(
            name='TLP',
            root_dir=self.root_dir)

    def _construct_seq_dict(self, root_dir):
        # image and annotation paths
        anno_files = sorted(glob.glob(
            osp.join(root_dir, '*/groundtruth_rect.txt')))
        seq_dirs = [osp.dirname(f) for f in anno_files]
        seq_names = [osp.basename(d) for d in seq_dirs]

        # construct seq_dict
        seq_dict = {}
        for s, seq_name in enumerate(seq_names):
            img_files = sorted(glob.glob(
                osp.join(seq_dirs[s], 'img/*.jpg')))
            anno = np.loadtxt(anno_files[s], delimiter=',')

            # parse annotations
            frames, bboxes, losts = anno[:, 0], anno[:, 1:5], anno[:, 5]
            assert np.all(frames == np.arange(len(frames)) + 1)
            bboxes[:, 2:] = bboxes[:, :2] + bboxes[:, 2:] - 1

            # meta information
            seq_len = len(img_files)
            img0 = ops.read_image(img_files[0])
            meta = {
                'width': img0.shape[1],
                'height': img0.shape[0],
                'frame_num': seq_len,
                'target_num': 1,
                'total_instances': seq_len,
                'absence': losts}
            
            # update seq_dict
            seq_dict[seq_name] = {
                'img_files': img_files,
                'target': {
                    'anno': bboxes,
                    'meta': meta}}

        return seq_dict
