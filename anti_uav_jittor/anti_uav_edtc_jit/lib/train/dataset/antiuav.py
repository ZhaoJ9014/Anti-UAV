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
import json
import cv2

class AntiUAV(BaseVideoDataset):
    """ AntiUAV dataset.
    """

    def __init__(self, root=None, image_loader=jpeg4py_loader, vid_ids=None, split=None, data_fraction=None):
        """
        args:
            root - path to the antiuav dataset.
            image_loader (jpeg4py_loader) -  The function to read the images. jpeg4py (https://github.com/ajkxyz/jpeg4py)
                                            is used by default.
            vid_ids - List containing the ids of the videos (1 - 20) used for training. If vid_ids = [1, 3, 5], then the
                    videos with subscripts -1, -3, and -5 from each class will be used for training.
            split - If split='train', the official train split (protocol-II) is used for training. Note: Only one of
                    vid_ids or split option can be used at a time.
            data_fraction - Fraction of dataset to be used. The complete dataset is used by default
        """
        root = env_settings().lasot_dir if root is None else root
        super().__init__('AntiUAV', root, image_loader)
        if split == 'train':
            self.root = os.path.join(root, 'train')
        elif split == 'val':
            self.root = os.path.join(root, 'validation')
        else:
            raise ValueError('Unknown split name.')

        # Keep a list of all classes
        self.sequence_list = self._get_sequence_list()

        if data_fraction is not None:
            self.sequence_list = random.sample(self.sequence_list, int(len(self.sequence_list)*data_fraction))

    def _get_sequence_list(self):
        with open(os.path.join(self.root, 'list.txt')) as f:
            dir_list = list(csv.reader(f))
        dir_list = [dir_name[0] for dir_name in dir_list]
        return dir_list

    def _build_class_list(self):
        seq_per_class = {}
        for seq_id, seq_name in enumerate(self.sequence_list):
            class_name = seq_name.split('-')[0]
            if class_name in seq_per_class:
                seq_per_class[class_name].append(seq_id)
            else:
                seq_per_class[class_name] = [seq_id]

        return seq_per_class

    def get_name(self):
        return 'antiuav'

    def has_class_info(self):
        return True

    def has_occlusion_info(self):
        return True

    def get_num_sequences(self):
        return len(self.sequence_list)

    # def get_num_classes(self):
    #     return len(self.class_list)

    def get_sequences_in_class(self, class_name):
        return self.seq_per_class[class_name]

    def _read_bb_anno(self, seq_path):
        bb_anno_file = os.path.join(seq_path, "IR_label.json")
        with open(bb_anno_file, 'r') as f:
            label_res = json.load(f)
        gt = label_res['gt_rect']
        if ([] in gt) or ([0] in gt):
            for i in range(len(gt)):
                if (gt[i]==[]) or (gt[i]==[0]):
                    gt[i] = [0,0,0,0]
        return jt.var(gt)

    def _read_target_visible(self, seq_path):
        # Read full occlusion and out_of_view
        exist_file = os.path.join(seq_path, "IR_label.json")
        with open(exist_file, 'r') as f:
            label_res = json.load(f)

        exist = label_res['exist']

        target_visible = jt.var(exist).byte()

        return target_visible

    def _get_sequence_path(self, seq_id):
        seq_name = self.sequence_list[seq_id]
        # class_name = seq_name.split('-')[0]
        # vid_id = seq_name.split('-')[1]

        return os.path.join(self.root,seq_name)

    def get_sequence_info(self, seq_id):
        seq_path = self._get_sequence_path(seq_id)
        bbox = self._read_bb_anno(seq_path)

        valid = (bbox[:, 2] > 0) & (bbox[:, 3] > 0)
        visible = self._read_target_visible(seq_path) & valid.byte()

        return {'bbox': bbox, 'valid': valid, 'visible': visible}

    def _get_frame_path(self, seq_path, frame_id):
        return os.path.join(seq_path, '{:06}.jpg'.format(frame_id+1))    # frames start from 1

    def _get_frame(self, seq_path, frame_id):
        return self.image_loader(self._get_frame_path(seq_path, frame_id))

    def _get_class(self, seq_path):
        raw_class = seq_path.split('/')[-2]
        return raw_class

    def get_class_name(self, seq_id):
        seq_path = self._get_sequence_path(seq_id)
        obj_class = self._get_class(seq_path)

        return obj_class

    def get_frames(self, seq_id, frame_ids, anno=None):
        seq_path = self._get_sequence_path(seq_id)

        obj_class = self._get_class(seq_path)
        frame_list = [self._get_frame(seq_path, f_id) for f_id in frame_ids]

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

    def get_UAVs(self, sequence):
        # sample all the target patch
        UAV_pathes = []
        print(sequence)
        sequence_path = os.path.join(self.root, sequence)
        anno_res = self._read_bb_anno(sequence_path)
        frame_list = [frame for frame in os.listdir(sequence_path) if frame.endswith(".jpg")]
        frame_list.sort(key=lambda f: int(f[:-4]))
        frames_list = [os.path.join(sequence_path, frame) for frame in frame_list]
        num_frame = len(frames_list)
        for i in range(num_frame):
            if anno_res[i][2] != 0 and anno_res[i][3] != 0:
                image = self.image_loader(frames_list[i])
                bbox = anno_res[i]
                uav_patch = self.sample_target(image, bbox, output_sz=16)
                UAV_pathes.append(uav_patch)

        return np.array(UAV_pathes)

    def sample_target(self, im, target_bb, output_sz=16):
        """ Extracts a square crop centered at target_bb box, of area search_area_factor^2 times target_bb area

        args:
            im - cv image
            target_bb - target box [x, y, w, h]
            search_area_factor - Ratio of crop size to target size
            output_sz - (float) Size to which the extracted crop is resized (always square). If None, no resizing is done.

        returns:
            cv image - extracted crop
            float - the factor by which the crop has been resized to make the crop size equal output_size
        """
        if not isinstance(target_bb, list):
            x, y, w, h = target_bb.tolist()
        else:
            x, y, w, h = target_bb

        x1 = x
        x2 = int(x1 + w)

        y1 = y
        y2 = int(y1 + h)

        # Crop target
        im_crop = im[y1 : y2, x1 : x2, :]
        target_patch = cv2.resize(im_crop, (output_sz, output_sz))

        return target_patch