import os
import os.path as osp
import glob
import numpy as np
import json
import hashlib

import libs.ops as ops
from libs.config import registry
from .dataset import SeqDataset


__all__ = ['VOT']


@registry.register_module
class VOT(SeqDataset):
    r"""`VOT <http://www.votchallenge.net/>`_ Datasets.

    Publication:
        ``The Visual Object Tracking VOT2017 challenge results``, M. Kristan, A. Leonardis
            and J. Matas, etc. 2017.
    
    Args:
        root_dir (string): Root directory of dataset where sequence
            folders exist.
        version (integer, optional): Specify the benchmark version. Specify as
            one of 2013~2018. Default is 2017.
        anno_type (string, optional): Returned annotation types, chosen as one of
            ``rect`` and ``corner``. Default is ``rect``.
        download (boolean, optional): If True, downloads the dataset from the internet
            and puts it in root directory. If dataset is downloaded, it is not
            downloaded again.
        list_file (string, optional): If provided, only read sequences
            specified by the file.
    """
    __valid_versions = [2013, 2014, 2015, 2016, 2017, 2018, 'LT2018',
                        2019, 'LT2019', 'RGBD2019', 'RGBT2019']

    def __init__(self, root_dir=None, version=2019, anno_type='rect',
                 download=True, list_file=None):
        assert version in self.__valid_versions, 'Unsupport VOT version.'
        assert anno_type in ['default', 'rect', 'inner_rect'], \
            'Unknown annotation type.'
        if root_dir is None:
            root_dir = osp.expanduser('~/data/vot{}'.format(version))
        self.root_dir = root_dir
        self.version = version
        self.anno_type = anno_type
        if download:
            self._download(root_dir, version)
        if list_file is None:
            list_file = osp.join(root_dir, 'list.txt')
        
        # initialize the dataset
        super(VOT, self).__init__(
            name='VOT-{}'.format(self.version),
            root_dir=self.root_dir,
            list_file=list_file)

    def _construct_seq_dict(self, root_dir, list_file):
        # image and annotation paths
        with open(list_file, 'r') as f:
            seq_names = f.read().strip().split('\n')
        seq_dirs = [osp.join(root_dir, s) for s in seq_names]
        anno_files = [osp.join(s, 'groundtruth.txt')
                           for s in seq_dirs]
        
        # construct seq_dict
        seq_dict = {}
        for s, seq_name in enumerate(seq_names):
            img_files = sorted(glob.glob(
                osp.join(seq_dirs[s], '*.jpg')))
            anno = np.loadtxt(anno_files[s], delimiter=',')
            anno = self._format(anno)

            # meta information
            seq_len = len(img_files)
            img0 = ops.read_image(img_files[0])
            meta = self._fetch_meta(seq_dirs[s], seq_len)
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

    def _download(self, root_dir, version):
        assert version in self.__valid_versions

        if not osp.isdir(root_dir):
            os.makedirs(root_dir)
        elif osp.isfile(osp.join(root_dir, 'list.txt')):
            with open(osp.join(root_dir, 'list.txt')) as f:
                seq_names = f.read().strip().split('\n')
            if all([osp.isdir(osp.join(root_dir, s)) for s in seq_names]):
                ops.sys_print('Files already downloaded.')
                return

        url = 'http://data.votchallenge.net/'
        if version in range(2013, 2015 + 1):
            # main challenge (2013~2015)
            homepage = url + 'vot{}/dataset/'.format(version)
        elif version in range(2015, 2019 + 1):
            # main challenge (2016~2019)
            homepage = url + 'vot{}/main/'.format(version)
        elif version.startswith('LT'):
            # long-term tracking challenge
            year = int(version[2:])
            homepage = url + 'vot{}/longterm/'.format(year)
        elif version.startswith('RGBD'):
            # RGBD tracking challenge
            year = int(version[4:])
            homepage = url + 'vot{}/rgbd/'.format(year)
        elif version.startswith('RGBT'):
            # RGBT tracking challenge
            year = int(version[4:])
            url = url + 'vot{}/rgbtir/'.format(year)
            homepage = url + 'meta/'
        
        # download description file
        bundle_url = homepage + 'description.json'
        bundle_file = osp.join(root_dir, 'description.json')
        if not osp.isfile(bundle_file):
            ops.sys_print('Downloading description file...')
            ops.download(bundle_url, bundle_file)

        # read description file
        ops.sys_print('\nParsing description file...')
        with open(bundle_file) as f:
            bundle = json.load(f)

        # md5 generator
        def md5(filename):
            hash_md5 = hashlib.md5()
            with open(filename, 'rb') as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
            return hash_md5.hexdigest()
        
        # download all sequences
        seq_names = []
        for seq in bundle['sequences']:
            seq_name = seq['name']
            seq_names.append(seq_name)

            # download channel (color/depth/ir) files
            channels = seq['channels'].keys()
            seq_files = []
            for cn in channels:
                seq_url = seq['channels'][cn]['url']
                if not seq_url.startswith(('http', 'https')):
                    seq_url = url + seq_url[seq_url.find('sequence'):]
                seq_file = osp.join(
                    root_dir,
                    '{}_{}.zip'.format(seq_name, cn))
                if not osp.isfile(seq_file) or \
                    md5(seq_file) != seq['channels'][cn]['checksum']:
                    ops.sys_print('\nDownloading %s...' % seq_name)
                    ops.download(seq_url, seq_file)
                seq_files.append(seq_file)

            # download annotations
            anno_url = homepage + '%s.zip' % seq_name
            anno_file = osp.join(root_dir, seq_name + '_anno.zip')
            if not osp.isfile(anno_file) or \
                md5(anno_file) != seq['annotations']['checksum']:
                ops.download(anno_url, anno_file)

            # unzip compressed files
            seq_dir = osp.join(root_dir, seq_name)
            if not osp.isfile(seq_dir) or len(os.listdir(seq_dir)) < 10:
                ops.sys_print('\nExtracting %s...' % seq_name)
                os.makedirs(seq_dir)
                for seq_file in seq_files:
                    ops.extract(seq_file, seq_dir)
                ops.extract(anno_file, seq_dir)

        # save list.txt
        list_file = osp.join(root_dir, 'list.txt')
        with open(list_file, 'w') as f:
            f.write(str.join('\n', seq_names))

        return root_dir
    
    def _format(self, anno):
        if anno.shape[1] == 8:
            if self.anno_type == 'rect':
                anno = self._corner2rect(anno)
            elif self.anno_type == 'inner_rect':
                anno = self._corner2rect_inner(anno)
        
        if anno.shape[1] == 4:
            anno[:, 2:] = anno[:, :2] + anno[:, 2:] - 1
        
        return anno

    def _corner2rect(self, corners, center=False):
        x1 = np.min(corners[:, 0::2], axis=1)
        x2 = np.max(corners[:, 0::2], axis=1)
        y1 = np.min(corners[:, 1::2], axis=1)
        y2 = np.max(corners[:, 1::2], axis=1)

        w = x2 - x1
        h = y2 - y1

        if center:
            cx = np.mean(corners[:, 0::2], axis=1)
            cy = np.mean(corners[:, 1::2], axis=1)
            return np.array([cx, cy, w, h]).T
        else:
            return np.array([x1, y1, w, h]).T

    def _corner2rect_inner(self, corners, center=False):
        cx = np.mean(corners[:, 0::2], axis=1)
        cy = np.mean(corners[:, 1::2], axis=1)

        x1 = np.min(corners[:, 0::2], axis=1)
        x2 = np.max(corners[:, 0::2], axis=1)
        y1 = np.min(corners[:, 1::2], axis=1)
        y2 = np.max(corners[:, 1::2], axis=1)

        area1 = np.linalg.norm(corners[:, 0:2] - corners[:, 2:4], axis=1) * \
            np.linalg.norm(corners[:, 2:4] - corners[:, 4:6], axis=1)
        area2 = (x2 - x1) * (y2 - y1)
        scale = np.sqrt(area1 / area2)
        w = scale * (x2 - x1) + 1
        h = scale * (y2 - y1) + 1

        if center:
            return np.array([cx, cy, w, h]).T
        else:
            return np.array([cx - w / 2, cy - h / 2, w, h]).T

    def _fetch_meta(self, seq_dir, frame_num):
        meta = {}

        # attributes
        tag_files = glob.glob(osp.join(seq_dir, '*.label')) + \
            glob.glob(osp.join(seq_dir, '*.tag'))
        for f in tag_files:
            if not osp.exists(f):
                continue
            tag = osp.basename(f)
            tag = tag[:tag.rfind('.')]
            meta[tag] = np.loadtxt(f)
        
        # practical
        practical_file = osp.join(seq_dir, 'practical')
        if osp.isfile(practical_file + '.value'):
            meta['practical'] = np.loadtxt(practical_file + '.value')
        if osp.isfile(practical_file + '.txt'):
            meta['practical_txt'] = np.loadtxt(practical_file + '.txt')

        # pad zeros if necessary
        for tag, val in meta.items():
            if len(val) < frame_num:
                meta[tag] = np.pad(
                    val, (0, frame_num - len(val)), 'constant')

        return meta
