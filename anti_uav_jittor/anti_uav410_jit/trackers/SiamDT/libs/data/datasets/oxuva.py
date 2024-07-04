import os.path as osp
import numpy as np

import libs.ops as ops
from libs.config import registry
from .dataset import SeqDataset


__all__ = ['OxUvA']


@registry.register_module
class OxUvA(SeqDataset):
    """`OxUvA <https://oxuva.github.io/long-term-tracking-benchmark/>`_ Dataset.

    Publication:
        ``Long-term Tracking in the Wild: a Benchmark``,
        J. Valmadre, L. Bertinetto, J. F. Henriques, ECCV 2015.
    
    Args:
        root_dir (string): Root directory of dataset where sequence
            folders exist.
        subset (string, optional): Specify ``dev`` or ``test`` subset of OxUvA.
        frame_stride (int, optional): Frame stride for down-sampling frames.
    """
    def __init__(self, root_dir=None, subset='dev', frame_stride=30):
        assert subset in ['dev', 'test']
        if root_dir is None:
            root_dir = osp.expanduser('~/data/OxUvA')
        self.root_dir = root_dir
        self.subset = subset
        self.frame_stride = frame_stride

        # initialize the dataset
        super(OxUvA, self).__init__(
            name='OxUvA_{}'.format(subset),
            root_dir=self.root_dir,
            subset=self.subset)

    def _construct_seq_dict(self, root_dir, subset):
        # load task information
        task_file = osp.join(root_dir, 'tasks/{}.csv'.format(subset))
        task = np.loadtxt(task_file, delimiter=',', dtype=str)

        # load dev annotations
        if subset == 'dev':
            dev_anno_file = osp.join(root_dir, 'annotations/dev.csv')
            dev_anno = np.loadtxt(dev_anno_file, delimiter=',', dtype=str)

        # construct seq_dict
        seq_dict = {}
        for s, line in enumerate(task):
            # parse task information
            vid_id, obj_id = line[:2]
            init_frame, last_frame = line[2:4].astype(int)
            init_anno = line[4:8].astype(np.float32)

            # log information
            seq_name = '_'.join([vid_id, obj_id])
            if s % 50 == 0 or s + 1 == len(task):
                ops.sys_print('Processing sequence [%d/%d]: %s...' % (
                    s + 1, len(task), seq_name))
            
            # parse annotations
            seq_dir = osp.join(root_dir, 'images', subset, vid_id)
            img0 = ops.read_image(seq_dir + '/000000.jpeg')
            h, w = img0.shape[:2]
            meta = {
                'width': img0.shape[1],
                'height': img0.shape[0],
                'target_num': 1}
            
            # parse and rescale initial annotations
            anno = np.expand_dims(init_anno[[0, 2, 1, 3]], axis=0)
            anno[:, [0, 2]] *= w
            anno[:, [1, 3]] *= h

            # image paths
            frames = np.arange(
                init_frame, last_frame + 1, self.frame_stride)
            img_files = [osp.join(seq_dir, '%06d.jpeg' % f)
                         for f in frames]
            
            # update meta information
            meta.update({
                'frame_num': len(img_files),
                'total_instances': len(img_files),
                'frames': frames})
            
            # update seq_dict
            seq_dict[seq_name] = {
                'img_files': img_files,
                'target': {
                    'anno': anno,
                    'meta': meta}}

        return seq_dict
