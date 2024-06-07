import os
import cv2
import json
import numpy as np
import sys
from glob import glob
from tqdm import tqdm
from PIL import Image

from pysot_toolkit.toolkit.datasets.dataset import  Dataset
import cv2


class Light_video(object):
    # def __init__(self,video_id,video_path,video_gt,confidece,model,video_gt_real):
    def __init__(self,video_id,video_path,model,video_gt_real):

        self.video_id = video_id
        self.video_path = video_path
        # self.video_gt = video_gt
        # self.confidence  = confidece
        self.model = model
        self.video_gt_real = video_gt_real
        list_imagename = [os.path.join(video_path, self.model,x) for x in os.listdir(os.path.join(self.video_path,self.model))]
        self.image_names = sorted(list_imagename,key=lambda x:int(x.split('{}I'.format(model))[-1].strip('.jpg')))

    def __getitem__(self,idx):
        return cv2.imread(self.image_names[idx]) ,self.video_gt_real[idx]

    def __len__(self):
        return len(self.image_names)

    def get_size(self):
        img = np.array(Image.open(self.image_names[0]), np.uint8)
        width = img.shape[1]
        height = img.shape[0]
        return width,height



def read(number,model,root,name=None):
    data_root = root
    gt = []
    confidence = []
    if name:
        path_video_name = os.path.join(data_root, name)
        json_path = os.path.join(path_video_name,'{}.json'.format(model))
        with open(json_path,'r')as f:
            metdata = json.load(f)
            con = metdata['confidence']
            confidence.append(con)
            gt_rect = metdata['bbox']
            gt.append(gt_rect)
            return gt, confidence
    for i,video_name in enumerate(os.listdir(data_root)[:number]):
        path_video_name = os.path.join(data_root,video_name)
        json_path = os.path.join(path_video_name,'{}.json'.format(model))
        with open(json_path,'r')as f :
            metdata = json.load(f)
            con = metdata['confidence']
            confidence.append(con)
            gt_rect = metdata['bbox']
            gt.append(gt_rect)
    # print(len(confidence))
    return gt,confidence


def read_real(number,model,root,name=None):
    data_root = root
    gt = []
    if name:
        path_video_name = os.path.join(data_root, name)
        json_path = os.path.join(path_video_name,'{}.json'.format(model))
        with open(json_path,'r')as f:
            metdata = json.load(f)
            gt_rect = metdata['gt_rect']
            gt.append(gt_rect)
            return gt
    for i,video_name in enumerate(os.listdir(data_root)[:number]):
        path_video_name = os.path.join(data_root,video_name)
        json_path = os.path.join(path_video_name,'{}.json'.format(model))
        print(json_path)
        with open(json_path) as f:
            metdata = json.load(f)
            gt_rect = metdata['gt_rect']
            gt.append(gt_rect)
    
    return gt

class multivideo:
    def __init__(self,ir,rgb):
        self.ir = ir
        self.rgb = rgb
    def __len__(self):
        return 1


class AntiFusion(Dataset):
    def __init__(self,number,name=None):
        super(Dataset, self).__init__()
        # self.dataset_root = 'D:\study\\track\dataset\Anti-UAV-RGBT\\train'
        # self.json_root = 'D:\study\\track\dataset\Anti-UAV-RGBT\\new_json'
        self.dataset_root = '/data01/xjy/code/modal/data/Anti_UAV_RGBT/test'
        # self.json_root = '/data01/xjy/code/modal/data/Anti_UAV_RGBT/label_new/test.json'
        ir_gt_real = read_real(number, root = self.dataset_root, model='infrared',name=name)
        rgb_gt_real = read_real(number, root = self.dataset_root, model='visible',name=name)
        self.name = 'UAVdataset'
        # self.videos = {}
        # number = len(os.listdir(self.dataset_root))
        # ir_gt,ir_con = read(number,root = self.json_root,model='infrared',name=name)
        # rgb_gt,rgb_con = read(number,root = self.json_root,model='visible',name=name)
        video_name = os.listdir(self.dataset_root)
        self.ir_videos = []
        self.rgb_videos = []
        if name:
            self.ir_videos.append(Light_video(
                video_id=name,
                # video_gt=ir_gt[0],
                # confidece=ir_con[0],
                video_gt_real=ir_gt_real[0],
                video_path=os.path.join(self.dataset_root,name),
                model='infrared'
            ))
            self.rgb_videos.append(Light_video(
                video_id=name,
                # video_gt=rgb_gt[0],
                # confidece=rgb_con[0],
                video_gt_real=rgb_gt_real[0],
                video_path=os.path.join(self.dataset_root, name),
                model='visible'
            ))
        else:
            for i in range(number):
                self.ir_videos.append(Light_video(
                    video_id= video_name[i],
                    # video_gt = ir_gt[i],
                    # confidece= ir_con[i],
                    video_gt_real=ir_gt_real[i],
                    video_path = os.path.join(self.dataset_root,str(video_name[i])),
                    model='infrared'
                ))
                self.rgb_videos.append(Light_video(
                    video_id= video_name[i],
                    # video_gt = rgb_gt[i],
                    # confidece = rgb_con[i],
                    video_gt_real=rgb_gt_real[i],
                    video_path = os.path.join(self.dataset_root,str(video_name[i])),
                    model='visible'
                ))

    def __getitem__(self, idx):
        return [self.rgb_videos[idx], self.ir_videos[idx]]

    def __len__(self):
        return len(self.ir_videos)

    def __iter__(self):
        len_ = len(self.ir_videos)
        # keys = sorted(list(self.ir_videos.keys()))
        for i in range(len_):
            yield self.ir_videos[i], self.rgb_videos[i]
