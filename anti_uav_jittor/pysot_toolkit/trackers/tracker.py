from __future__ import absolute_import

import numpy as np
import math
import torchvision.transforms.functional as tvisf
import cv2


import jittor as jt
from jittor import transform
import time
from dectect.dect_predict import FRCNN
from PIL import Image
class Tracker(object):

    def __init__(self, name, net, window_penalty=0.49, exemplar_size=128, instance_size=256):
        self.name = name
        self.net = net
        self.window_penalty = window_penalty
        self.exemplar_size = exemplar_size
        self.instance_size = instance_size
        self.net = self.net
        self.dector_ir = FRCNN(mode='ir')
        self.dector_rgb = FRCNN(mode='rgb')
        self.threholds = {'confidence': 0.55,
                          'dis_modals':0.3,
                          }

    def _convert_score(self, score):

        score = score.permute(2, 1, 0).contiguous().view(2, -1).permute(1, 0)
        score = jt.nn.softmax(score, dim=1).data[:, 0]
        return score

    def _convert_bbox(self, delta):

        delta = delta.permute(2, 1, 0).contiguous().view(4, -1)
        delta = delta.data

        return delta

    def _bbox_clip(self, cx, cy, width, height, boundary):
        cx = max(0, min(cx, boundary[1]))
        cy = max(0, min(cy, boundary[0]))
        width = max(10, min(width, boundary[1]))
        height = max(10, min(height, boundary[0]))
        return cx, cy, width, height

    def get_subwindow(self, im, pos, model_sz, original_sz, avg_chans):
        """
        args:
            im: rgb based image
            pos: center position
            model_sz: exemplar size
            original_sz: original size
            avg_chans: channel average
        """
        if isinstance(pos, float):
            pos = [pos, pos]
        sz = original_sz
        im_sz = im.shape
        c = (original_sz + 1) / 2
        # context_xmin = round(pos[0] - c) # py2 and py3 round
        context_xmin = np.floor(pos[0] - c + 0.5)
        context_xmax = context_xmin + sz - 1
        # context_ymin = round(pos[1] - c)
        context_ymin = np.floor(pos[1] - c + 0.5)
        context_ymax = context_ymin + sz - 1
        left_pad = int(max(0., -context_xmin))
        top_pad = int(max(0., -context_ymin))
        right_pad = int(max(0., context_xmax - im_sz[1] + 1))
        bottom_pad = int(max(0., context_ymax - im_sz[0] + 1))

        context_xmin = context_xmin + left_pad
        context_xmax = context_xmax + left_pad
        context_ymin = context_ymin + top_pad
        context_ymax = context_ymax + top_pad

        r, c, k = im.shape
        if any([top_pad, bottom_pad, left_pad, right_pad]):
            size = (r + top_pad + bottom_pad, c + left_pad + right_pad, k)
            te_im = np.zeros(size, np.uint8)
            te_im[top_pad:top_pad + r, left_pad:left_pad + c, :] = im
            if top_pad:
                te_im[0:top_pad, left_pad:left_pad + c, :] = avg_chans
            if bottom_pad:
                te_im[r + top_pad:, left_pad:left_pad + c, :] = avg_chans
            if left_pad:
                te_im[:, 0:left_pad, :] = avg_chans
            if right_pad:
                te_im[:, c + left_pad:, :] = avg_chans
            im_patch = te_im[int(context_ymin):int(context_ymax + 1),
                       int(context_xmin):int(context_xmax + 1), :]
        else:
            im_patch = im[int(context_ymin):int(context_ymax + 1),
                       int(context_xmin):int(context_xmax + 1), :]

        if not np.array_equal(model_sz, original_sz):
            im_patch = cv2.resize(im_patch, (model_sz, model_sz))
        im_patch = im_patch.transpose(2, 0, 1)
        im_patch = im_patch[np.newaxis, :, :, :]
        im_patch = im_patch.astype(np.float32)
        im_patch = jt.array(im_patch)
        im_patch = im_patch
        return im_patch




    def initialize_features(self):
        if not getattr(self, 'features_initialized', False):
            self.net.initialize()
        self.features_initialized = True

    def initialize(self, image, info: dict,model) -> dict:
        tic = time.time()
        hanning = np.hanning(32)
        window = np.outer(hanning, hanning)
        self.window = window.flatten()
        # Initialize
        self.initialize_features()
        self.pre_score = 1
        bbox = info['init_bbox']
        center_pos = np.array([bbox[0] + bbox[2] / 2,
                                    bbox[1] + bbox[3] / 2])

        size = np.array([bbox[2], bbox[3]])

        # calculate z crop size
        w_z = size[0] + (2 - 1) * ((size[0] + size[1]) * 0.5)
        h_z = size[1] + (2 - 1) * ((size[0] + size[1]) * 0.5)
        s_z = math.ceil(math.sqrt(w_z * h_z))

        # calculate channle average
        self.channel_average = np.mean(image, axis=(0, 1))

        # get crop
        z_crop = self.get_subwindow(image, center_pos,
                                    self.exemplar_size,
                                    s_z, self.channel_average)

        # normalize
        z_crop = z_crop.float().mul(1.0 / 255.0).clamp(0.0, 1.0)
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        self.inplace = False
        # z_crop[0] = tvisf.normalize(z_crop[0], self.mean, self.std, self.inplace)
        if model == 'rgb':
            self.center_pos_rgb = center_pos
            self.size_rgb = size
            self.or_rgb = size
        else:
            self.center_pos_ir = center_pos
            self.size_ir = size
            self.or_ir = size
        return z_crop


    def multi_ininize(self,images,rgb_info, ir_info):
        image_ir = images['ir']
        image_rgb = images['rgb']
        z_crop_ir = self.initialize(image_ir,ir_info,model='ir')
        z_crop_rgb = self.initialize(image_rgb, rgb_info,model='rgb')
        template = {'ir':z_crop_ir,'rgb':z_crop_rgb}
        self.net.template(template)



    def track(self, images):
        # calculate x crop size
        image_ir = images['ir']
        image_rgb = images['rgb']
        if sum(self.size_ir) == 0:
            self.size_ir = self.or_ir
        if sum(self.size_rgb) == 0:
            self.size_rgb = self.or_rgb
        w_x_ir = self.size_ir[0] + (5 - 1) * ((self.size_ir[0] + self.size_ir[1]) * 0.5)
        h_x_ir = self.size_ir[1] + (5 - 1) * ((self.size_ir[0] + self.size_ir[1]) * 0.5)
        s_x_ir = math.ceil(math.sqrt(w_x_ir * h_x_ir))
        self.s_x_ir = s_x_ir

        w_x_rgb = self.size_rgb[0] + (5 - 1) * ((self.size_rgb[0] + self.size_rgb[1]) * 0.5)
        h_x_rgb = self.size_rgb[1] + (5 - 1) * ((self.size_rgb[0] + self.size_rgb[1]) * 0.5)
        s_x_rgb = math.ceil(math.sqrt(w_x_rgb * h_x_rgb))
        self.s_x_rgb = s_x_rgb
        # get crop
        # if self.pre_score > 0.7:
        # if s_x_ir==0 or s_x_rgb==0:
        #     out = {'target_bbox': [[0, 0, 0, 0]],
        #            'best_score': 0.7
        #            }
        #     return out,out
        x_crop_ir = self.get_subwindow(image_ir, self.center_pos_ir,
                                self.instance_size,
                                round(s_x_ir), self.channel_average)

        x_crop_rgb = self.get_subwindow(image_rgb, self.center_pos_rgb,
                                self.instance_size,
                                round(s_x_rgb), self.channel_average)

        # normalize
        x_crop_ir = x_crop_ir.float().mul(1.0 / 255.0).clamp(0.0, 1.0)
        # x_crop_ir[0] = tvisf.normalize(x_crop_ir[0], self.mean, self.std, self.inplace)
        x_crop_ir[0] = transform.image_normalize(x_crop_ir[0], self.mean, self.std)
        x_crop_rgb = x_crop_rgb.float().mul(1.0 / 255.0).clamp(0.0, 1.0)
        # x_crop_rgb[0] = tvisf.normalize(x_crop_rgb[0], self.mean, self.std, self.inplace)

        # track

        outputs = self.net.track(x_crop_ir,x_crop_rgb)
        rgb_output, ir_output = outputs[-1]
        output_ir =  self.finnal_out(ir_output,mode='ir',image=image_ir)
        output_rgb = self.finnal_out(rgb_output,mode='rgb',image=image_rgb)
        return  output_ir,output_rgb



    def finnal_out(self,out,mode,image):
        score = self._convert_score(out['pred_logits'])
        pred_bbox = self._convert_bbox(out['pred_boxes'])

        # window penalty
        pscore = score * (1 - self.window_penalty) + \
                 self.window * self.window_penalty

        best_idx = np.argmax(pscore)
        self.pre_score = pscore[best_idx]
        bbox = pred_bbox[:, best_idx]
        if mode == 'rgb':
            center_pos = self.center_pos_rgb
            s_x = self.s_x_rgb
        else:
            center_pos = self.center_pos_ir
            s_x = self.s_x_ir
        bbox = bbox * s_x
        cx = bbox[0] + center_pos[0] - s_x / 2
        cy = bbox[1] + center_pos[1] - s_x / 2
        width = bbox[2]
        height = bbox[3]

        # clip boundary
        cx, cy, width, height = self._bbox_clip(cx, cy, width,
                                                height, image.shape[:2])

        # update state
        # self.center_pos = np.array([cx, cy])
        # self.size = np.array([width, height])

        bbox = [cx - width / 2,
                cy - height / 2,
                width,
                height]

        fout = {'target_bbox': [bbox],
               'best_score': pscore[best_idx]}
        return fout



    def re_dectect(self,image,mode):
        w = 3
        h = 1080 / 512
        if mode == 'ir':
            image = Image.fromarray(image)
            top_conf, top_boxes = self.dector_ir.detect_image(image, crop=False, count=False)
            w = 1
            h = 1
        else:
            image = cv2.resize(image,(640,512))
            image = Image.fromarray(image)
            top_conf, top_boxes = self.dector_rgb.detect_image(image, crop=False, count=False)

        if top_conf[0] == 0:
            out = {'target_bbox': [0, 0, 0, 0],
                   'best_score': 0
                   }
            self.former_zero = True
            return out

        for i in range(len(top_conf)):
            top, left, bottom, right = top_boxes[i]
            cy = (top + bottom) / 2
            cx = (left + right) / 2
            width, height = math.fabs(left - right), math.fabs(top - bottom)
            top_boxes[i] = [(cx - width / 2) * w,
                    (cy - height / 2) * h,
                    width * w,
                    height * h]
        best_id = np.argmax(top_conf)
        out = {'target_bbox': top_boxes,
            'best_score':top_conf[best_id]
        }
        return out


    def track_multi_modal(self,images):
        x_factor = 3
        y_factor = 1080/512
        ir_redect,rgb_redect = False,False

        image_ir = images['ir']
        image_rgb = images['rgb']
        resutl_ir,resutl_rgb = self.track(images)
        # resutl_rgb = self.re_dectect(image_rgb, mode='rgb')
        # rgb_redect = True
        wrgb,hrgb,_ = image_rgb.shape
        wir, hir, _ = image_ir.shape
        ##以IR为主要
        bbox_ir  = resutl_ir['target_bbox'][0]
        bbox_rgb = resutl_rgb['target_bbox'][0]


        center_ir = np.array([bbox_ir[0] + bbox_ir[2]/2, bbox_ir[1] + bbox_ir[3]/2])
        center_rgb = np.array([bbox_rgb[0] + bbox_rgb[2] / 2, bbox_rgb[1] + bbox_rgb[3] / 2])
        distance = np.linalg.norm(center_ir/np.array([hir,wir]) - center_rgb/np.array([hrgb,wrgb]))
        # print(center_ir,center_rgb,distance)
        # print(center_ir,sum(center_ir / np.array([hir, wir])))
        distance_ir_self = np.linalg.norm(center_ir/np.array([hir,wir]) - np.array([0.5,0.5]))

        if distance_ir_self > 0.45 :
            resutl_ir = self.re_dectect(image_ir, mode='ir')
        if sum(center_rgb/np.array([hrgb,wrgb])) < 0.4 :

            resutl_rgb = self.re_dectect(image_rgb, mode='rgb')
        # resutl_ir['best_score'] = 0.2
        # resutl_rgb['best_score'] = 0.2



        if resutl_ir['best_score'] < self.threholds['confidence']:
            # print('confidence,ir{}'.format(resutl_ir['best_score']),'confidence,rgb{}'.format(resutl_rgb['best_score']))
            resutl_ir = self.re_dectect(image_ir,mode='ir')
            ir_redect = True


        if resutl_rgb['best_score'] < self.threholds['confidence']:
            resutl_rgb = self.re_dectect(image_rgb,mode='rgb')
            rgb_redect = True
        if distance > self.threholds['dis_modals']:
            if rgb_redect == True:
                pass
            else:
                if resutl_rgb['best_score'] < resutl_ir['best_score']:
                    resutl_rgb = self.re_dectect(image_rgb,mode='rgb')
                    rgb_redect =True

            if ir_redect == True:
                pass
            else:
                if resutl_rgb['best_score'] > resutl_ir['best_score']:
                    resutl_ir = self.re_dectect(image_ir,mode='ir')
                    ir_redect =True

        size_ir = np.array([bbox_ir[2], bbox_ir[3]])
        size_rgb = np.array([bbox_rgb[2], bbox_rgb[3]])/np.array([x_factor,y_factor])
        size_ir = size_ir[0]*size_ir[1]
        size_rgb = size_rgb[0]*size_rgb[1]
        # print(size_ir/size_rgb)

        if size_rgb/(size_ir+0.1) > 3:
            if not rgb_redect:
                resutl_rgb = self.re_dectect(image_rgb,mode='rgb')


        distance = 10
        ir_final_id = 0
        rgb_final_id = 0
        if ir_redect or rgb_redect:
            for id_ir , box_ir in enumerate(resutl_ir['target_bbox']):
                if isinstance(box_ir,int):
                    break
                center_ir = np.array([box_ir[0] + box_ir[2] / 2, box_ir[1] + box_ir[3] / 2])
            for id_rgb , box_rgb in enumerate(resutl_rgb['target_bbox']):
                if isinstance(box_rgb,int):
                    break
                center_rgb = np.array([box_rgb[0] + box_rgb[2] / 2, box_rgb[1] + box_rgb[3] / 2])
                dis = np.linalg.norm(center_ir / np.array([hir, wir]) - center_rgb / np.array([hrgb, wrgb]))
                if dis < distance:
                    distance = dis
                    ir_final_id = id_ir
                    rgb_final_id = id_rgb


        result_ir_bbox = resutl_ir['target_bbox'][ir_final_id]
        result_rgb_bbox = resutl_rgb['target_bbox'][rgb_final_id]



        if isinstance(result_ir_bbox,int) or isinstance(result_rgb_bbox,int): #其中一个为空
            if not isinstance(result_rgb_bbox,int): # rgb不为空，ir为空
                self.center_pos_ir = np.array(
                    [(result_rgb_bbox[0] + result_rgb_bbox[2] / 2)/x_factor, (result_rgb_bbox[1] + result_rgb_bbox[3] / 2)/y_factor])
                result_ir_bbox = [0,0,0,0]
            else:
                if not isinstance(result_ir_bbox, int): # rgb 为空 ， ir不为空
                    self.center_pos_rgb = np.array(
                        [(result_ir_bbox[0] + result_ir_bbox[2] / 2) * x_factor,
                         (result_ir_bbox[1] + result_ir_bbox[3] / 2) * y_factor])
                    result_rgb_bbox = [0, 0, 0, 0]
                else: # rgb 为空 ， ir为空
                    self.center_pos_rgb = np.array([hrgb / 2, wrgb / 2])
                    self.center_pos_ir = np.array([hir / 2, hrgb / 2])
                    result_ir_bbox = [0, 0, 0, 0]
                    result_rgb_bbox = [0, 0, 0, 0]

        else:
            self.center_pos_ir =  np.array([result_ir_bbox[0] + result_ir_bbox[2] /2, result_ir_bbox[1] + result_ir_bbox[3] /2])
            self.size_ir = np.array([result_ir_bbox[2], result_ir_bbox[3]])
            self.center_pos_rgb = np.array([result_rgb_bbox[0] + result_rgb_bbox[2] / 2, result_rgb_bbox[1] + result_rgb_bbox[3] / 2])
            self.size_rgb = np.array([result_rgb_bbox[2], result_rgb_bbox[3]])

        return {'ir':result_ir_bbox,'rgb':result_rgb_bbox}

    def track_only_local(self,images):
        image_ir = images['ir']
        image_rgb = images['rgb']
        # resutl_ir = self.re_dectect(image_rgb, mode='rgb')
        # resutl_rgb = self.re_dectect(image_rgb, mode='ir')
        resutl_ir,resutl_rgb = self.track(images)
        result_ir_bbox = resutl_ir['target_bbox'][0]
        result_rgb_bbox = resutl_rgb['target_bbox'][0]
        self.center_pos_ir =  np.array([result_ir_bbox[0] + result_ir_bbox[2] /2, result_ir_bbox[1] + result_ir_bbox[3] /2])
        self.size_ir = np.array([result_ir_bbox[2], result_ir_bbox[3]])
        self.center_pos_rgb = np.array([result_rgb_bbox[0] + result_rgb_bbox[2] / 2, result_rgb_bbox[1] + result_rgb_bbox[3] / 2])
        self.size_rgb = np.array([result_rgb_bbox[2], result_rgb_bbox[3]])
        return {'ir': result_ir_bbox, 'rgb': result_rgb_bbox}


