import os.path as osp
import glob
import numpy as np
import json

import libs.ops as ops
from libs.config import registry
from .dataset import SeqDataset


__all__ = ['UAV123']


@registry.register_module
class UAV123(SeqDataset):
    """`UAV123 <https://ivul.kaust.edu.sa/Pages/pub-benchmark-simulator-uav.aspx>`_ Dataset.

    Publication:
        ``A Benchmark and Simulator for UAV Tracking``,
        M. Mueller, N. Smith and B. Ghanem, ECCV 2016.
    
    Args:
        root_dir (string): Root directory of dataset where sequence
            folders exist.
        version (integer or string): Specify the benchmark version, specify as one of
            ``UAV123`` and ``UAV20L``.
    """
    def __init__(self, root_dir=None, version='UAV123'):
        assert version.upper() in ['UAV20L', 'UAV123']
        if root_dir is None:
            root_dir = osp.expanduser('~/data/UAV123')
        self.root_dir = root_dir
        self.version = version.upper()

        # initialize the dataset
        super(UAV123, self).__init__(
            name=self.version,
            root_dir=self.root_dir,
            version=self.version)
    
    def _construct_seq_dict(self, root_dir, version):
        # sequence meta information
        meta_file = osp.join(
            osp.dirname(__file__), 'uav123.json')
        with open(meta_file) as f:
            seq_metas = json.load(f)

        # image and annotation paths
        anno_files = sorted(glob.glob(
            osp.join(root_dir, 'anno/%s/*.txt' % version)))
        seq_names = [
            osp.basename(f)[:-4] for f in anno_files]
        seq_dirs = [osp.join(
            root_dir, 'data_seq/UAV123/%s' % \
                seq_metas[version][n]['folder_name'])
            for n in seq_names]
        
        # construct seq_dict
        seq_dict = {}
        for s, seq_name in enumerate(seq_names):
            # valid frame range
            start_frame = seq_metas[version][
                seq_names[s]]['start_frame']
            end_frame = seq_metas[version][
                seq_names[s]]['end_frame']
            img_files = [osp.join(
                seq_dirs[s], '%06d.jpg' % f)
                for f in range(start_frame, end_frame + 1)]

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
