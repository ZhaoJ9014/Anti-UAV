import numpy as np
from lib.test.evaluation.data import Sequence, BaseDataset, SequenceList
from lib.test.utils.load_text import load_text
import os
import json

class AntiUAVDataset(BaseDataset):
    """
        AntiUAV dataset.
    """
    def __init__(self):
        super().__init__()
        self.base_path = self.env_settings.antiuav_path
        self.sequence_list = self._get_sequence_list()

    def get_sequence_list(self):
        return SequenceList([self._construct_sequence(s) for s in self.sequence_list])

    def _construct_sequence(self, sequence_name):

        anno_path = '{}/{}/IR_label.json'.format(self.base_path, sequence_name)
        with open(anno_path, 'r') as f:
            label_res = json.load(f)
        gt = label_res['gt_rect']
        if ([] in gt) or ([0] in gt):
            for i in range(len(gt)):
                if (gt[i]==[]) or (gt[i]==[0]):
                    gt[i] = [0,0,0,0]

        ground_truth_rect = np.array(gt)
        frames_path = '{}/{}'.format(self.base_path, sequence_name)
        frame_list = [frame for frame in os.listdir(frames_path) if frame.endswith(".jpg")]
        frame_list.sort(key=lambda f: int(f[:-4]))
        frames_list = [os.path.join(frames_path, frame) for frame in frame_list]

        return Sequence(sequence_name, frames_list, 'antiuav', ground_truth_rect.reshape(-1, 4))

    def __len__(self):
        return len(self.sequence_list)

    def _get_sequence_list(self):
        with open('{}/list.txt'.format(self.base_path)) as f:
            sequence_list = f.read().splitlines()
        return sequence_list
