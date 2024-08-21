import os
import os.path
import jittor as jt
import numpy as np
import pandas
import csv
import random
from collections import OrderedDict
from .base_video_dataset import BaseVideoDataset
from lib.train.data import jpeg4py_loader
from lib.train.admin import env_settings


class TNL2k(BaseVideoDataset):
    def __init__(self, root=None, image_loader=jpeg4py_loader, vid_ids=None, split=None, data_fraction=None):
        """
        args:
            root - path to the lasot dataset.
            image_loader (jpeg4py_loader) -  The function to read the images. jpeg4py (https://github.com/ajkxyz/jpeg4py)
                                            is used by default.
            vid_ids - List containing the ids of the videos (1 - 20) used for training. If vid_ids = [1, 3, 5], then the
                    videos with subscripts -1, -3, and -5 from each class will be used for training.
            split - If split='train', the official train split (protocol-II) is used for training. Note: Only one of
                    vid_ids or split option can be used at a time.
            data_fraction - Fraction of dataset to be used. The complete dataset is used by default
        """
        root = env_settings().tnl2k_dir if root is None else root
        super().__init__('TNL2k', root, image_loader)

        # Keep a list of all classes
        # self.class_list = [f for f in os.listdir(self.root)]
        # self.class_to_id = {cls_name: cls_id for cls_id, cls_name in enumerate(self.class_list)}

        self.sequence_list = self._build_sequence_list(vid_ids, split)

        if data_fraction is not None:
            self.sequence_list = random.sample(self.sequence_list, int(len(self.sequence_list)*data_fraction))

        # self.seq_per_class = self._build_class_list()

    def _build_sequence_list(self, vid_ids=None, split=None):
        assert split == 'train'
        if split == 'train':
            split_name = 'TNL2K_train_subset'
        self.dataset_split_path = os.path.join(self.root, split_name)

        # ltr_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..')
        # file_path = os.path.join(ltr_path, 'data_specs', 'lasot_train_split.txt')
        # sequence_list = pandas.read_csv(file_path, header=None, squeeze=True).values.tolist()
        sequence_list = os.listdir(self.dataset_split_path)
        seq_imgs = {}
        for sequence in sequence_list:
            if not os.path.isdir(os.path.join(self.dataset_split_path, sequence)):
                sequence_list.remove(sequence)
                continue

            imgs = sorted(os.listdir(os.path.join(self.dataset_split_path, sequence, 'imgs')))
            seq_imgs[sequence] = imgs
        self.seq_imgs = seq_imgs

        return sequence_list

    # def _build_class_list(self):
    #     seq_per_class = {}
    #     for seq_id, seq_name in enumerate(self.sequence_list):
    #         class_name = seq_name.split('-')[0]
    #         if class_name in seq_per_class:
    #             seq_per_class[class_name].append(seq_id)
    #         else:
    #             seq_per_class[class_name] = [seq_id]
    #
    #     return seq_per_class

    def get_name(self):
        return 'tnl2k'

    def has_class_info(self):
        return True

    def has_occlusion_info(self):
        return True

    def get_num_sequences(self):
        return len(self.sequence_list)

    def get_num_classes(self):
        # return len(self.class_list)
        return 0

    def get_sequences_in_class(self, class_name):
        # return self.seq_per_class[class_name]
        return None

    def _read_bb_anno(self, seq_path):
        bb_anno_file = os.path.join(seq_path, "groundtruth.txt")
        gt = pandas.read_csv(bb_anno_file, delimiter=',', header=None, dtype=np.float32, na_filter=False, low_memory=False).values
        return jt.var(gt)

    # def _read_target_visible(self, seq_path):
    #     # Read full occlusion and out_of_view
    #     occlusion_file = os.path.join(seq_path, "full_occlusion.txt")
    #     out_of_view_file = os.path.join(seq_path, "out_of_view.txt")
    #
    #     with open(occlusion_file, 'r', newline='') as f:
    #         occlusion = torch.ByteTensor([int(v) for v in list(csv.reader(f))[0]])
    #     with open(out_of_view_file, 'r') as f:
    #         out_of_view = torch.ByteTensor([int(v) for v in list(csv.reader(f))[0]])
    #
    #     target_visible = ~occlusion & ~out_of_view
    #
    #     return target_visible

    def _get_sequence_path(self, seq_id):
        seq_name = self.sequence_list[seq_id]
        # class_name = seq_name.split('-')[0]
        # vid_id = seq_name.split('-')[1]

        return os.path.join(self.dataset_split_path, seq_name), seq_name

    def get_sequence_info(self, seq_id):
        seq_path, seq_name = self._get_sequence_path(seq_id)
        bbox = self._read_bb_anno(seq_path)

        valid = (bbox[:, 2] > 0) & (bbox[:, 3] > 0)
        visible = [True] * len(valid)
        # visible = self._read_target_visible(seq_path) & valid.byte()

        return {'bbox': bbox, 'valid': valid, 'visible': jt.Var(visible)}

    def _get_frame_path(self, seq_name, frame_id):
        img_path = os.path.join(self.dataset_split_path, seq_name, 'imgs', self.seq_imgs[seq_name][frame_id])
        # print(img_path)

        return img_path

    def _get_frame(self, seq_name, frame_id):
        return self.image_loader(self._get_frame_path(seq_name, frame_id))

    # def _get_class(self, seq_path):
    #     raw_class = seq_path.split('/')[-2]
    #     return raw_class

    # def get_class_name(self, seq_id):
    #     seq_path = self._get_sequence_path(seq_id)
    #     obj_class = self._get_class(seq_path)
    #
    #     return obj_class

    def get_frames(self, seq_id, frame_ids, anno=None):
        seq_path, seq_name = self._get_sequence_path(seq_id)

        obj_class = "padding"
        frame_list = [self._get_frame(seq_name, f_id) for f_id in frame_ids]

        if anno is None:
            anno = self.get_sequence_info(seq_id)

        anno_frames = {}
        for key, value in anno.items():
            anno_frames[key] = [value[f_id, ...].clone() for f_id in frame_ids]

        object_meta = OrderedDict({'object_class_name': obj_class,
                                   'motion_class': None,
                                   'major_class': None,
                                   'root_class': None,
                                   'motion_adverb': None})

        return frame_list, anno_frames, object_meta
