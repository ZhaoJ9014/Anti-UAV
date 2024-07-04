import os
import os.path as osp
import glob
import numpy as np
import io
from itertools import chain

import libs.ops as ops
from libs.config import registry
from .dataset import SeqDataset


__all__ = ['OTB']


@registry.register_module
class OTB(SeqDataset):
    r"""`OTB <http://cvlab.hanyang.ac.kr/tracker_benchmark/>`_ Datasets.

    Publication:
        ``Object Tracking Benchmark``, Y. Wu, J. Lim and M.-H. Yang, IEEE TPAMI 2015.
    
    Args:
        root_dir (string): Root directory of dataset where sequence
            folders exist.
        version (integer or string): Specify the benchmark version, specify as one of
            ``2013``, ``2015``, ``tb50`` and ``tb100``.
        download (boolean, optional): If True, downloads the dataset from the internet
            and puts it in root directory. If dataset is downloaded, it is not
            downloaded again.
    """
    __otb13_seqs = ['Basketball', 'Bolt', 'Boy', 'Car4', 'CarDark',
                    'CarScale', 'Coke', 'Couple', 'Crossing', 'David',
                    'David2', 'David3', 'Deer', 'Dog1', 'Doll', 'Dudek',
                    'FaceOcc1', 'FaceOcc2', 'Fish', 'FleetFace',
                    'Football', 'Football1', 'Freeman1', 'Freeman3',
                    'Freeman4', 'Girl', 'Ironman', 'Jogging', 'Jumping',
                    'Lemming', 'Liquor', 'Matrix', 'Mhyang', 'MotorRolling',
                    'MountainBike', 'Shaking', 'Singer1', 'Singer2',
                    'Skating1', 'Skiing', 'Soccer', 'Subway', 'Suv',
                    'Sylvester', 'Tiger1', 'Tiger2', 'Trellis', 'Walking',
                    'Walking2', 'Woman']

    __tb50_seqs = ['Basketball', 'Biker', 'Bird1', 'BlurBody', 'BlurCar2',
                   'BlurFace', 'BlurOwl', 'Bolt', 'Box', 'Car1', 'Car4',
                   'CarDark', 'CarScale', 'ClifBar', 'Couple', 'Crowds',
                   'David', 'Deer', 'Diving', 'DragonBaby', 'Dudek',
                   'Football', 'Freeman4', 'Girl', 'Human3', 'Human4',
                   'Human6', 'Human9', 'Ironman', 'Jump', 'Jumping',
                   'Liquor', 'Matrix', 'MotorRolling', 'Panda', 'RedTeam',
                   'Shaking', 'Singer2', 'Skating1', 'Skating2', 'Skiing',
                   'Soccer', 'Surfer', 'Sylvester', 'Tiger2', 'Trellis',
                   'Walking', 'Walking2', 'Woman']

    __tb100_seqs = ['Bird2', 'BlurCar1', 'BlurCar3', 'BlurCar4', 'Board',
                    'Bolt2', 'Boy', 'Car2', 'Car24', 'Coke', 'Coupon',
                    'Crossing', 'Dancer', 'Dancer2', 'David2', 'David3',
                    'Dog', 'Dog1', 'Doll', 'FaceOcc1', 'FaceOcc2', 'Fish',
                    'FleetFace', 'Football1', 'Freeman1', 'Freeman3',
                    'Girl2', 'Gym', 'Human2', 'Human5', 'Human7', 'Human8',
                    'Jogging', 'KiteSurf', 'Lemming', 'Man', 'Mhyang',
                    'MountainBike', 'Rubik', 'Singer1', 'Skater',
                    'Skater2', 'Subway', 'Suv', 'Tiger1', 'Toy', 'Trans',
                    'Twinnings', 'Vase'] + __tb50_seqs

    __otb15_seqs = __tb100_seqs

    __version_dict = {
        2013: __otb13_seqs,
        2015: __otb15_seqs,
        50: __tb50_seqs,
        100: __tb100_seqs}

    def __init__(self, root_dir=None, version=2015, download=True):
        assert version in self.__version_dict
        if root_dir is None:
            root_dir = osp.expanduser('~/data/OTB')
        self.root_dir = root_dir
        self.version = version
        if download:
            self._download(root_dir, version)
        
        # initialize the dataset
        super(OTB, self).__init__(
            name='OTB-{}'.format(self.version),
            root_dir=self.root_dir,
            version=self.version)

    def _construct_seq_dict(self, root_dir, version):
        # image and annotation paths
        valid_seqs = self.__version_dict[version]
        anno_files = sorted(list(chain.from_iterable(glob.glob(
            osp.join(root_dir, s, 'groundtruth*.txt')) for s in valid_seqs)))
        # remove empty annotation files
        # (e.g., groundtruth_rect.1.txt of Human4)
        anno_files = self._filter_files(anno_files)
        seq_dirs = [osp.dirname(f) for f in anno_files]
        seq_names = [osp.basename(d) for d in seq_dirs]
        # rename repeated sequence names
        # (e.g., Jogging and Skating2)
        seq_names = self._rename_seqs(seq_names)

        # construct seq_dict
        seq_dict = {}
        for s, seq_name in enumerate(seq_names):
            img_files = sorted(glob.glob(
                osp.join(seq_dirs[s], 'img/*.jpg')))
            
            # special sequences
            # (visit http://cvlab.hanyang.ac.kr/tracker_benchmark/index.html for detail)
            if seq_name.lower() == 'david':
                img_files = img_files[300-1:770]
            elif seq_name.lower() == 'football1':
                img_files = img_files[:74]
            elif seq_name.lower() == 'freeman3':
                img_files = img_files[:460]
            elif seq_name.lower() == 'freeman4':
                img_files = img_files[:283]
            elif seq_name.lower() == 'diving':
                img_files = img_files[:215]
            
            # load annotations (to deal with different delimeters)
            with open(anno_files[s], 'r') as f:
                anno = np.loadtxt(io.StringIO(f.read().replace(',', ' ')))
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

    def _download(self, root_dir, version):
        assert version in self.__version_dict
        seq_names = self.__version_dict[version]

        if not osp.isdir(root_dir):
            os.makedirs(root_dir)
        elif all([osp.isdir(osp.join(root_dir, s)) for s in seq_names]):
            ops.sys_print('Files already downloaded.')
            return

        url_fmt = 'http://cvlab.hanyang.ac.kr/tracker_benchmark/seq/%s.zip'
        for seq_name in seq_names:
            seq_dir = osp.join(root_dir, seq_name)
            if osp.isdir(seq_dir):
                continue
            url = url_fmt % seq_name
            zip_file = osp.join(root_dir, seq_name + '.zip')
            ops.sys_print('Downloading to %s...' % zip_file)
            ops.download(url, zip_file)
            ops.sys_print('\nExtracting to %s...' % root_dir)
            ops.extract(zip_file, root_dir)

        return root_dir
