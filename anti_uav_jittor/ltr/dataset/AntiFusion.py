import os
import os.path
import numpy as np
from .base_video_dataset import BaseVideoDataset
from ltr.data.image_loader import jpeg4py_loader_w_failsafe
from ltr.admin.environment import env_settings
import json
import cv2
import jittor as jt
class AntiUav_fusion(BaseVideoDataset):
    def __init__(self, root=None, image_loader=jpeg4py_loader_w_failsafe, split=None, seq_ids=None, data_fraction=None):
        root = env_settings().got10k_dir if root is None else root
        super().__init__('AntiRGBT', root, image_loader)
        self.sequence_list = self._get_sequence_list()

    def get_name(self):
        return 'AntiRGBT'

    def has_class_info(self):
        return False

    def has_occlusion_info(self):
        return True

    def _get_sequence_list(self):
        return os.listdir(self.root)

    def _load_meta_info(self):
        sequence_meta_info = {s: self._read_meta(os.path.join(self.root, s)) for s in self.sequence_list}
        return sequence_meta_info

    def _get_sequence_path(self, seq_id):
        return os.path.join(self.root, self.sequence_list[seq_id])

    def get_sequence_info(self, seq_id):
        '''
            function : load the anna data of the seq
            arg:
                seq_id : the id of the seq
            return :
                info : which contain the RGB frames and IR frames
        '''
        info = {}
        seq_path = self._get_sequence_path(seq_id)
        bb_infrared_anno_file = os.path.join(seq_path, 'infrared.json')
        bb_visible_anno_file = os.path.join(seq_path, 'visible.json')
        # print(bb_visible_anno_file)
        info['RGB'] = self._get_info_from_json(bb_visible_anno_file,mode= 'rgb')
        info['IR'] = self._get_info_from_json(bb_infrared_anno_file,mode='ir')
        info['visible'] = info['RGB']['visible'] * info['IR']['visible']
        return info

    def _get_info_from_json(self, path,mode):
        '''
            fuction: from json file read data
            arg:
                path: json file path
            return:
                info : a dict whose keys contain visible, bbox, valid
        '''
        # info = {}
        # with open(path, 'r') as f:
        #     metdata = json.load(f)
        #     info['visible'] = jt.array(metdata['exist'],dtype=jt.float32)
        #     index = jt.Var(info['visible']) > 0
        #     zero_pad = jt.zeros((len(info['visible']), 4), dtype=jt.float32)
        #     visible_data = np.array(metdata['gt_rect'], dtype=object)[index]
        #     zero_pad[index] = jt.array(visible_data,  dtype = jt.float32)
        #     info['bbox'] = zero_pad
        #     if mode == 'rgb':
        #         scale_width = 640 / 1920
        #         scale_height = 512 / 1080
        #         zero_pad[:, 0] *= scale_width  # x1
        #         zero_pad[:, 1] *= scale_height  # y1
        #         zero_pad[:, 2] *= scale_width  # x2
        #         zero_pad[:, 3] *= scale_height  # y2
        #     bbox = info['bbox']
        #     info['valid'] = (bbox[:, 2] > 0) & (bbox[:, 3] > 0)
        # return info
        info = {}
        with open(path, 'r') as f:
            metdata = json.load(f)
            info['visible'] = jt.Var(metdata['exist'])
            index = jt.Var(info['visible']) > 0
            zero_pad = jt.zeros((len(info['visible']), 4), dtype=jt.float32)
            visible_data = np.array(metdata['gt_rect'], dtype=object)[index].tolist()
            zero_pad[index] = jt.array(visible_data , dtype = jt.float32)
            info['bbox'] = zero_pad
            if mode == 'rgb':
                scale_width = 640 / 1920
                scale_height = 512 / 1080
                zero_pad[:, 0] *= scale_width  # x1
                zero_pad[:, 1] *= scale_height  # y1
                zero_pad[:, 2] *= scale_width  # x2
                zero_pad[:, 3] *= scale_height  # y2
            bbox = info['bbox']
            info['valid'] = (bbox[:, 2] > 0) & (bbox[:, 3] > 0)
        return info

    def _get_frame(self, seq_path, frame_id):
        im = self.image_loader(self._get_frame_path(seq_path, frame_id))
        im = cv2.resize(im, (640, 512))
        return im

    def _get_frame_path(self, seq_path, frame_id):
        model = os.path.split(seq_path)[-1]
        frame_name = model + 'I'+'{}'.format(frame_id).zfill(4) + '.jpg'
        return os.path.join(seq_path, frame_name)

    def get_frames(self, seq_id, frame_ids, anno=None):
        '''
            according to the id of the seq and the frame obtain the pairs of the frame
            arg:
                seq_id : the id of the seq
                frame_ids : a list which contains which frames you want
            return :
                frame_list : the pairs list of IR frames and RGB frames
                anno_frames: the pairs list of IR annotation and RGB annotation
                info : the info of the video
        '''
        seq_path = self._get_sequence_path(seq_id)
        info = self.get_sequence_info(seq_id)
        frame_list = {}
        frame_list['RGB'] = [self._get_frame(os.path.join(seq_path,'visible'), f_id) for f_id in frame_ids]
        frame_list['IR'] = [self._get_frame(os.path.join(seq_path,'infrared'), f_id) for f_id in frame_ids]
        anno_frames = {'RGB': {}, 'IR': {}}
        if anno is None:
            anno = self.get_sequence_info(seq_id)
        # print(anno['RGB']['visible'])
        for key, value in anno['RGB'].items():
            anno_frames['RGB'][key] = [value[f_id, ...].clone() for f_id in frame_ids]

        for key, value in anno['IR'].items():
            anno_frames['IR'][key] = [value[f_id, ...].clone() for f_id in frame_ids]
        return frame_list, anno_frames, info