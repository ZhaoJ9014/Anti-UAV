import os
import os.path as osp
import glob
import numpy as np
import io
import json
from itertools import chain

import libs.ops as ops
from libs.config import registry
from .dataset import SeqDataset


__all__ = ['UAVtir']


@registry.register_module
class UAVtir(SeqDataset):

    r"""`UAVtir`_ Datasets.

       Publication:


       Args:
           root_dir (string): Root directory of dataset where sequence
               folders exist.

       """

    def __init__(self, root_dir=None, subset='test', download=True):

        assert subset in ['train', 'val', 'test']

        if root_dir is None:
            root_dir = osp.expanduser('~/data/~')
        self.root_dir = root_dir

        self.subset = subset

        self.name = 'UAVtir_{}'.format(subset)

        # initialize the dataset
        super(UAVtir, self).__init__(
            name=self.name,
            root_dir=self.root_dir,
            subset=self.subset)

    def _construct_seq_dict(self, root_dir, subset):

        newpath=os.path.join(root_dir,subset)
        # image and annotation paths
        anno_files = sorted(glob.glob(os.path.join(newpath,
                                                        '*/IR_label.json')))
        seq_dirs = [osp.dirname(f) for f in anno_files]
        seq_names = [osp.basename(d) for d in seq_dirs]

        # construct seq_dict
        seq_dict = {}
        for s, seq_name in enumerate(seq_names):

            img_files = sorted(glob.glob(
                osp.join(seq_dirs[s], '*.jpg')))

            # load annotations (to deal with different delimeters)

            with open(anno_files[s], 'r') as f:


                anno = json.load(f)['gt_rect']

                # import pdb;pdb.set_trace()

                anno=np.array(anno,dtype=np.float32)
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

    def _filter_files(self, filenames):
        filtered_files = []
        for filename in filenames:
            with open(filename, 'r') as f:
                if f.read().strip() == '':
                    ops.sys_print('Warning: %s is empty.' % filename)
                else:
                    filtered_files.append(filename)

        return filtered_files

    def _rename_seqs(self, seq_names):
        # in case some sequences may have multiple targets
        renamed_seqs = []
        for i, seq_name in enumerate(seq_names):
            if seq_names.count(seq_name) == 1:
                renamed_seqs.append(seq_name)
            else:
                ind = seq_names[:i + 1].count(seq_name)
                renamed_seqs.append('%s.%d' % (seq_name, ind))

        return renamed_seqs

    def _download(self, root_dir):
        if not osp.isdir(root_dir):
            os.makedirs(root_dir)
        elif len(os.listdir(root_dir)) > 100:
            ops.sys_print('Files already downloaded.')
            return

        url = 'http://XXX.zip'
        zip_file = osp.join(root_dir, 'XXX.zip')
        ops.sys_print('Downloading to %s...' % zip_file)
        ops.download(url, zip_file)
        ops.sys_print('\nExtracting to %s...' % root_dir)
        ops.extract(zip_file, root_dir)

        return root_dir
