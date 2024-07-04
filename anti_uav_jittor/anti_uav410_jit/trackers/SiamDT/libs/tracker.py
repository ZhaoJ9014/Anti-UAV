import numpy as np
import time

import libs.ops as ops
from .model import Model

__all__ = ['Tracker', 'OxUvA_Tracker']


class Tracker(Model):

    def __init__(self, name, is_deterministic=True, visualize=False,
                 input_type='image', color_fmt='RGB'):
        assert input_type in ['image', 'file']
        assert color_fmt in ['RGB', 'BGR', 'GRAY']
        super(Tracker, self).__init__()
        self.name = name
        self.is_deterministic = is_deterministic
        self.input_type = input_type
        self.color_fmt = color_fmt
        self.visualize = visualize
        if self.visualize:
            self.visini(5123)

    def init(self, img, init_bbox):
        raise NotImplementedError

    def update(self, img):
        raise NotImplementedError

    def visini(self, portnum):
        # python -m visdom.server -port=5123
        from visdom import Visdom
        self.viz = Visdom(port=portnum)
        assert self.viz.check_connection()

    def forward_test(self, img_files, init_bbox, visualize=False):
        # state variables
        frame_num = len(img_files)
        bboxes = np.zeros((frame_num, 4))
        bboxes[0] = init_bbox
        times = np.zeros(frame_num)

        for f, img_file in enumerate(img_files):
            if self.input_type == 'image':
                img = ops.read_image(img_file, self.color_fmt)
            elif self.input_type == 'file':
                img = img_file

            begin = time.time()
            if f == 0:
                self.init(img, init_bbox)
                up_flag = True
            else:
                current_box, up_flag = self.update(img)
                bboxes[f, :] = current_box
                # # 没有检测到目标，使用上一帧的结果
                # if len(current_box) == 0:
                #     bboxes[f, :]=bboxes[f-1, :]
                # else:
                #     bboxes[f, :] = current_box
            times[f] = time.time() - begin

            if self.visualize:
                ops.show_image(img, bboxes[f, :], f, up_flag, self.viz)

        return bboxes, times


class OxUvA_Tracker(Tracker):

    def update(self, img):
        r'''One needs to return (bbox, score, present) in
            function `update`.
        '''
        raise NotImplementedError

    def forward_test(self, img_files, init_bbox, visualize=False):
        frame_num = len(img_files)
        times = np.zeros(frame_num)
        preds = [{
            'present': True,
            'score': 1.0,
            'xmin': init_bbox[0],
            'xmax': init_bbox[2],
            'ymin': init_bbox[1],
            'ymax': init_bbox[3]}]

        for f, img_file in enumerate(img_files):
            if self.input_type == 'image':
                img = ops.read_image(img_file, self.color_fmt)
            elif self.input_type == 'file':
                img = img_file

            begin = time.time()
            if f == 0:
                self.init(img, init_bbox)
            else:
                bbox, score, present = self.update(img)
                preds.append({
                    'present': present,
                    'score': score,
                    'xmin': bbox[0],
                    'xmax': bbox[2],
                    'ymin': bbox[1],
                    'ymax': bbox[3]})
            times[f] = time.time() - begin

            if visualize:
                ops.show_image(img, bboxes[f, :])

        # update the preds as one-per-second
        frame_stride = 30
        preds = {f * frame_stride: pred for f, pred in enumerate(preds)}

        return preds, times
