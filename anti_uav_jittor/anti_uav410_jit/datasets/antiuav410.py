from __future__ import absolute_import, print_function, unicode_literals

import os
import glob
import numpy as np
import io
import six
from itertools import chain
import json

from utils.ioutils import download, extract


class AntiUAV410(object):
    r"""`AntiUAV410`_ Datasets.

    Publication:

    
    Args:
        root_dir (string): Root directory of dataset where sequence
            folders exist.

    """

    def __init__(self, root_dir, download=True):

        super(AntiUAV410, self).__init__()

        self.root_dir = root_dir

        # 文件个数超过100，不下载
        if download:
            self._download(root_dir)
        self._check_integrity(root_dir)

        self.anno_files = sorted(glob.glob(
            os.path.join(root_dir, '*/IR_label.json')))

        self.seq_dirs = [os.path.dirname(f) for f in self.anno_files]


        self.seq_names = [os.path.basename(d) for d in self.seq_dirs]


    def __getitem__(self, index):
        r"""        
        Args:
            index (integer or string): Index or name of a sequence.
        
        Returns:
            tuple: (img_files, anno), where ``img_files`` is a list of
                file names and ``anno`` is a N x 4 (rectangles) numpy array.
        """
        if isinstance(index, six.string_types):
            if not index in self.seq_names:
                raise Exception('Sequence {} not found.'.format(index))
            index = self.seq_names.index(index)

        img_files = sorted(glob.glob(
            os.path.join(self.seq_dirs[index], '*.jpg')))

        with open(self.anno_files[index], 'r') as f:
            label_res = json.load(f)

        assert len(img_files) == len(label_res['gt_rect'])
        assert len(label_res['gt_rect'][0]) == 4

        return img_files, label_res

    def __len__(self):
        return len(self.seq_names)

    def _filter_files(self, filenames):
        filtered_files = []
        for filename in filenames:
            with open(filename, 'r') as f:
                if f.read().strip() == '':
                    print('Warning: %s is empty.' % filename)
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
        if not os.path.isdir(root_dir):
            os.makedirs(root_dir)
        elif len(os.listdir(root_dir)) > 10:
            print('Files already downloaded.')
            return

        url = 'XXXX.zip'
        zip_file = os.path.join(root_dir, 'XXXX.zip')
        print('Downloading to %s...' % zip_file)
        download(url, zip_file)
        print('\nExtracting to %s...' % root_dir)
        extract(zip_file, root_dir)

        return root_dir

    def _check_integrity(self, root_dir):
        seq_names = os.listdir(root_dir)
        seq_names = [n for n in seq_names if not n[0] == '.']

        if os.path.isdir(root_dir) and len(seq_names) > 0:
            # check each sequence folder
            for seq_name in seq_names:
                seq_dir = os.path.join(root_dir, seq_name)
                if not os.path.isdir(seq_dir):
                    print('Warning: sequence %s not exists.' % seq_name)
        else:
            # dataset not exists
            raise Exception('Dataset not found or corrupted. ' +
                            'You can use download=True to download it.')
