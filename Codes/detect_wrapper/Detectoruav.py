import argparse
import os
import shutil
import time
from pathlib import Path
import numpy as np

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
from PIL import Image
import pdb

import sys
from detect_wrapper.models.experimental import attempt_load
from detect_wrapper.utils.datasets import LoadStreams, LoadImages
from detect_wrapper.utils.general import (
    check_img_size, non_max_suppression, apply_classifier, scale_coords,
    xyxy2xywh, plot_one_box, strip_optimizer, set_logging)
from detect_wrapper.utils.torch_utils import select_device, load_classifier, time_synchronized



class DroneDetection:
    def __init__(self, IRweights_path: str, RGBweights_path: str):
        '''
        initialize all detection
        '''
        #try:
        parser = argparse.ArgumentParser()
        parser.add_argument('--IRweights', nargs='+', type=str, default=IRweights_path, help='mdel.pt.path(s)') #'/home/dell/Project/Project_UAV_new/detect_wrapper/weights/best.pt'
        parser.add_argument('--RGBweights', nargs='+', type=str, default=RGBweights_path, help='mdel.pt.path(s)')
        parser.add_argument('--source', type=str, default='/home/dell/Project_UAV/detect_wrapper/inference', help='source')
        parser.add_argument('--image-size', type=int, default='640', help='inference size')
        parser.add_argument('--conf-thres', type=float, default='0.25', help='object confidence threshold')  #0.25 
        parser.add_argument('--iou-thres', type=float, default='0.45', help='IOU threshold for NMS')   #0.45
        parser.add_argument('--device', default='0', help='cuda device, i.e. 0,1,2,3 or cpu')
        parser.add_argument('--view-img', action='store_true', help='display results')
        parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
        parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
        parser.add_argument('--save-dir', type=str, default='inference/output', help='directory to save results')
        parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --clas 0 2 3')
        parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
        parser.add_argument('--augment', action='store_true', help='augumented inference')
        parser.add_argument('--update', action='store_true', help='update all models')
        self.opt = parser.parse_args()  
        
        self.device = select_device(self.opt.device)
        self.half = self.device.type != 'cpu' 
        #Load  IR model
        self.IR_model = attempt_load(self.opt.IRweights, map_location=self.device)
        self.IR_imgsz = check_img_size(self.opt.image_size, s=self.IR_model.stride.max())
        #Load  RGB model
        self.RGB_model = attempt_load(self.opt.RGBweights, map_location=self.device)
        self.RGB_imgsz = check_img_size(self.opt.image_size, s=self.RGB_model.stride.max())
        
        if self.half:
            self.IR_model.half()
            self.RGB_model.half() 
        self.IR_img = torch.zeros((1, 3, self.IR_imgsz, self.IR_imgsz), device=self.device)
        _ = self.IR_model(self.IR_img.half() if self.half else self.IR_img) if self.device.type != 'cpu' else None

        self.RGB_img = torch.zeros((1, 3, self.RGB_imgsz, self.RGB_imgsz), device=self.device)
        _ = self.RGB_model(self.RGB_img.half() if self.half else self.RGB_img) if self.device.type != 'cpu' else None

    def forward_IR(self, frame):
        '''
        detect drones in IR mode
        '''    
        img = cv2.resize(frame, (640, 512))
        img = img.transpose((2, 0, 1))
    
        img = torch.from_numpy(img).to(self.device)
        img = img.half() if self.half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        # pdb.set_trace()
        pred = self.IR_model(img, augment=self.opt.augment)[0]
        pred = non_max_suppression(pred, self.opt.conf_thres, self.opt.iou_thres, classes=self.opt.classes, agnostic=self.opt.agnostic_nms)[0]
        scale_pred = scale_coords(img.shape[2:], pred[:, :4], frame.shape).round() if pred is not None and len(pred) else None

        if scale_pred is not None:
            bbox = scale_pred[0].cpu().numpy().tolist()
            bbox = [bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]]
            return bbox
        else:
            return scale_pred

    def forward_RGB(self, frame):
        '''
        detect drones in RGB mode
        '''
        img = cv2.resize(frame, (640, 384))
        img = img.transpose((2, 0, 1))

        img = torch.from_numpy(img).to(self.device)
        img = img.half() if self.half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        pred = self.RGB_model(img, augment=self.opt.augment)[0]
        pred = non_max_suppression(pred, self.opt.conf_thres, self.opt.iou_thres, classes=self.opt.classes, agnostic=self.opt.agnostic_nms)[0]

        scale_pred = scale_coords(img.shape[2:], pred[:, :4], frame.shape).round() if pred is not None and len(pred) else None

        if scale_pred is not None:
            bbox = scale_pred[0].cpu().numpy().tolist()
            bbox = [bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]]
            
            return bbox
        else:
            return scale_pred

            
        
    


if __name__ == '__main__':
    droneDetect = DroneDetection()
