import os
import os.path
import numpy as np
import jittor as jt
import csv
import pandas
import random
from collections import OrderedDict
from .base_video_dataset import BaseVideoDataset
from ltr.data.image_loader import jpeg4py_loader
from ltr.admin.environment import env_settings
import json

class AntiUav(BaseVideoDataset):
    def __init__(self, root=None, image_loader=jpeg4py_loader, split=None, seq_ids=None, data_fraction=None):
        super().__init__('AntiUav', root, image_loader)
        self.sequence_list = self._get_sequence_list()

    def get_name(self):
        return 'AntiUav'

    def has_class_info(self):
        return False

    def has_occlusion_info(self):
        return True

    def _get_sequence_list(self):
        return os.listdir(self.root)
        # for i, video_name in enumerate(os.listdir(self.root)):
        #     path_video_name = os.path.join(self.root, video_name)
        #     path_video_name_list.append(path_video_name)
        # return  path_video_name_list
    def _load_meta_info(self):
        sequence_meta_info = {s: self._read_meta(os.path.join(self.root, s)) for s in self.sequence_list}
        return sequence_meta_info

    def _get_sequence_path(self,seq_id):
        return os.path.join(self.root,self.sequence_list[seq_id])



    def get_sequence_info(self, seq_id):
        info = {}
        seq_path = self._get_sequence_path(seq_id)
        bb_anno_file = os.path.join(seq_path, 'label.json')
        with open(bb_anno_file,'r')as f :
            metdata = json.load(f)
            info['visible'] = jt.var(metdata['exist'])
            info['bbox'] = jt.var(metdata['gt_rect'],dtype=jt.float32)
            bbox = info['bbox']
            info['valid']= (bbox[:, 2] > 0) & (bbox[:, 3] > 0)
        return info

    def _get_frame(self, seq_path, frame_id):
        return self.image_loader(self._get_frame_path(seq_path, frame_id))

    def _get_frame_path(self, seq_path, frame_id):
        return os.path.join(seq_path, '{}.jpg'.format(frame_id+1))

    def get_frames(self, seq_id, frame_ids, anno=None):
        seq_path = self._get_sequence_path(seq_id)
        info = self.get_sequence_info(seq_id)
        frame_list = [self._get_frame(seq_path, f_id) for f_id in frame_ids]
        anno_frames = {}
        if anno is None:
            anno = self.get_sequence_info(seq_id)
        for key, value in anno.items():
            index = [id for id in frame_ids]
            # print(key,value)
            anno_frames[key] = [value[f_id, ...].clone() for f_id in frame_ids]
        return frame_list, anno_frames, info




