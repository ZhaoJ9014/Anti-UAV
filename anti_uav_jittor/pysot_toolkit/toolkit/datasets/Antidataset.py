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
    def __init__(self,video_id,video_path,video_gt):
        self.video_id = video_id
        self.video_path = video_path
        self.video_gt = video_gt
        self.model = 'infrared'
        list_imagename = [os.path.join(video_path, self.model,x) for x in os.listdir(os.path.join(self.video_path,self.model))]
        self.image_names = sorted(list_imagename,key=lambda x:int(x.split('infraredI')[-1].strip('.jpg')))
    def __getitem__(self,idx):
        return cv2.imread(self.image_names[idx]) ,self.video_gt[idx]
    def get_size(self):
        img = np.array(Image.open(self.image_names[0]), np.uint8)
        width = img.shape[1]
        height = img.shape[0]
        return width,height


def read(number):
    data_root = 'D:\study\\track\dataset\Anti-UAV-RGBT\\train'
    gt = []
    for i,video_name in enumerate(os.listdir(data_root)):
        path_video_name = os.path.join(data_root,video_name)
        # video_path = os.path.join(path_video_name,'IR.mp4')
        json_path = os.path.join(path_video_name,'infrared.json')
        with open(json_path,'r')as f :
            metdata = json.load(f)
            frame_len = len(metdata['exist'])
            gt_rect = metdata['gt_rect']
            gt.append(gt_rect)
    return gt[:number]


class AntiDataset(Dataset):
    def __init__(self,number):
        super(Dataset, self).__init__()
        self.dataset_root = 'D:\study\\track\dataset\Anti-UAV-RGBT\\train'
        self.name = 'UAVdataset'
        self.videos = {}
        gt = read(number)
        video_name = os.listdir(self.dataset_root)
        for i in range(number):
            self.videos[str(i)] = Light_video(
                video_id= video_name[i],
                video_gt = gt[i],
                video_path = os.path.join(self.dataset_root,str(video_name[i]))
            )

# test_dataset = AntiDataset(100)
# x = test_dataset[16:]
# print(len(x))