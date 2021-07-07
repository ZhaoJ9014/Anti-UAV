from __future__ import division

import queue
import pdb
import os
import time
import datetime
import argparse
import struct
import socket
import json
import logging

import sys
sys.path.append(r"C:\Users\uavproject\Desktop\detection_sys")
from models import *
from utils.utils import *
from utils.datasets import *
import multiprocessing

from PIL import Image
import cv2

import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
import matplotlib.pyplot as plt

import matplotlib.patches as patches
from matplotlib.ticker import NullLocator

import kcftracker

import numpy as np
from multiprocessing import Process

def safe_log(msg):
    if g_logger:
        g_logger.info(msg)

def bg_filter(detection):
    #print(detection)
    coord = detection[:, :4]
    w = coord[:, 2] - coord[:, 0]
    h = coord[:, 3] - coord[:, 1]
    is_drone = (w * h) > 42
    #is_drone = (w > 8) * (h > 5)
    coord[:, 0] += w * 0
    coord[:, 1] += h * 0
    coord[:, 2] = w * 1
    coord[:, 3] = h * 1
    return coord[is_drone]


def test1():
        safe_log("detector gets !!!")



class DroneDetection():
    def __init__(self):
        try:
            parser = argparse.ArgumentParser()
            parser.add_argument("--model_def", type=str, default=r"C:\Users\uavproject\Desktop\detection_sys\config\yolov3-custom.cfg", help="path to model definition file")
            parser.add_argument("--weights_path", type=str, default=r"C:\Users\uavproject\Desktop\detection_sys\yolov3_ckpt_190_IR_BS25_SIZE608_.pth", help="path to weights file")
            parser.add_argument("--conf_thres", type=float, default=0.994, help="object confidence threshold")
            parser.add_argument("--nms_thres", type=float, default=0.4, help="iou thresshold for non-maximum suppression")
            parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
            opt = parser.parse_args()
            
            device_type = 'cuda' #"cuda" if torch.cuda.is_available() else "cpu"
            #only cuda
            device = torch.device(device_type)
            safe_log('device  : {} - {}'.format(device_type, device))
            self.device = device
            self.model = Darknet(opt.model_def, img_size=opt.img_size).cuda()
            self.model.load_state_dict(torch.load(opt.weights_path, map_location=device))
            self.model.eval()  
            self.model.to(device)
            pretest = transforms.ToTensor()(Image.open(r'C:\Users\uavproject\Desktop\detection_sys\test.jpg'))
            self.frame_H = pretest.size(1)
            self.frame_W = pretest.size(2)
            safe_log('{}'.format(pretest.size()))
            pretest, _ = pad_to_square(pretest, 0)
            pretest = resize(pretest, opt.img_size)
            self.pretest = pretest.unsqueeze(0).to(device)
            with torch.no_grad():
                self.model(self.pretest)
                safe_log("model is OK!") 
            self.x = 1   
            self.opt=opt
        except Exception as err:
            safe_log('!!! error happens 0 : {}'.format(err))
        
    def forward(self,frame):
        try:
            #safe_log('{}  -- {}'.format(self.model, self.pretest.size()))
            #self.model(self.pretest)
            opt = self.opt
            device = self.device
            frame_W = frame.shape[1]
            frame_H = frame.shape[0]  
            # img = Image.fromarray(frame)  
            img = Image.open(r'C:\Users\uavproject\Desktop\detection_sys\test.jpg')
            input_img = torchvision.transforms.ToTensor()(img)
            input_img, _ = pad_to_square(input_img, 0)
            input_img = resize(input_img, opt.img_size)
            input_img = input_img.unsqueeze(0)
            safe_log('unsqueeze done  cuda is {} {}'.format(torch.cuda.is_available(), input_img.size()))
            input_img = torch.tensor(input_img, device=device)  #input_img.to(device)
            safe_log('before model  {}  {}'.format(type(input_img), input_img.device))  
            with torch.no_grad():
                detections = self.model(input_img)

            safe_log('after model')
            detections = non_max_suppression(detections, opt.conf_thres, opt.nms_thres)
            if detections[0] is not None:
                detections = rescale_boxes(detections[0], opt.img_size, (frame_H, frame_W))
                detections = bg_filter(detections) 
                if torch.cuda.is_available():
                    detections = detections.cpu()
                detections = detections[0].numpy().tolist()
                detections = [int(x) for x in detections]
            boundingbox = detections
            return boundingbox
        except Exception as err:
            safe_log('!!! error happens 1 : {}'.format(err))
            return None
        # try:
        #     cv2.rectangle(frame,(boundingbox[0],boundingbox[1]), (boundingbox[0]+boundingbox[2],boundingbox[1]+boundingbox[3]), (0,0,255), 4)
        # except:
        #     print("error rectangle")
        # cv2.imshow('tracking',frame)  
    def test_detection(self):
        img = np.ones((512, 640, 3), dtype=np.uint8)
        safe_log("hahahah")
        z = cv2.resize(z, (416,416))
        safe_log("zzzzz")
        img = Image.fromarray(img)
        input_img = torchvision.transforms.ToTensor()(img)
        input_img, _ = pad_to_square(input_img, 0)
        input_img = resize(input_img, self.opt.img_size)
        input_img = input_img.unsqueeze(0)
        safe_log("create")
        device_cuda = torch.device("cuda:0")
        safe_log("before transfer")
        try:
            #img = torch.tensor(input_img, device=device_cuda)
            #print(img.is_cuda)
            img = input_img.to(device_cuda)
        except Exception as err:
            safe_log(' error happens 3 : {}'.format(err))
        #img = input_img.cuda()
        safe_log("after transfer")
        with torch.no_grad():
            safe_log("before model")
            print(img.size())
            detections = self.model(img)
            safe_log("after model")
         
    def test_frame(self, frame):
        mono_to_rgb(frame)
        safe_log("rectangle")
        cv2.rectangle(frame,(100,100), (200,200), (0,0,255), 4)
        safe_log("imread")
        cv2.imread("C:\data\img.png", frame)

        



class  DroneTracker():
    def __init__(self):
        self.tracker = kcftracker.KCFTracker(False, True, True)
        
    def init_track(self, init_box, init_frame):

        self.tracker.init(init_box, init_frame)
        
    def on_track(self, frame):
        boundingbox = self.tracker.update(frame)
        boundingbox = list(map(int, boundingbox))
        #cv2.rectangle(frame,(boundingbox[0],boundingbox[1]), (boundingbox[0]+boundingbox[2],boundingbox[1]+boundingbox[3]), (0,0,255), 4)
        #cv2.imshow('tracking', frame)
        #后续可以删除
        #cv2.waitKey(6)
        return boundingbox
    
    #测试方法
    def test_track(self, imgstr="111"):
        print("tracker gets {imgstr}")
        return [12,34,545,88]


def test():
    interval=50
    cap = cv2.VideoCapture(r'C:\Users\uavproject\Desktop\detection_sys\test1.mp4')
    ret, frame = cap.read()
    drone_det=DroneDetection()
    drone_tracker=DroneTracker()
    while(ret):
        init_box=drone_det.forward(frame)
        drone_tracker.init_track(init_box,frame)
        num=0
        while(num<interval):
            num=num+1
            ret, frame = cap.read()
            if ret:
                bbx=drone_tracker.on_track(frame)
    cap.release()
    print("done......")
        
g_init = False
g_detector = None  # 检测器
g_tracker = None   # 跟踪器
g_logger = None
g_enable_log = True
g_frame_counter = 0
TRACK_MAX_COUNT = 100
#创建子进程


def global_init():
    """ global variables initialize.
    """
    global g_init, g_detector, g_tracker, g_logger, g_enable_log
    if not g_init:
        if g_enable_log:
            g_logger = logging.getLogger()
            g_logger.setLevel(logging.INFO)
            fh = logging.FileHandler('c:/data/log.txt', mode='a')
            g_logger.addHandler(fh)
        g_detector = DroneDetection()
        g_tracker = DroneTracker()
        g_init = True
    safe_log("global init done")


def send_bbs(bbs):
    global g_logger
    if g_logger:
        g_logger.info('send a box : {}'.format(bbs))

def mono_to_rgb(data):
    w, h = data.shape
    img = np.zeros((w, h, 3), dtype=np.uint8)
    img[:, :, 0] = data
    img[:, :, 1] = data
    img[:, :, 2] = data
    return img


def recvimg1(data):
    global g_init, g_detector, g_tracker, g_frame_counter
    safe_log('recv data type {}'.format(type(data)))
    # np.save('frame_1.npy', data)
    frame = np.array(data)
    frame =  mono_to_rgb(frame) # 转格式
    safe_log('recv a frame')
    bbx = None
    if g_detector and g_tracker:
        ts = time.time()
        if g_frame_counter <= 0:
            #safe_log(' detector.forward')
            #time.sleep(50)
            safe_log('before detector.forward')
            init_box = g_detector.forward(frame)
            safe_log('after detector.forward')
            ##g_tracker.init_track(init_box, frame)
            ##safe_log('init done')
            ##g_frame_counter = TRACK_MAX_COUNT
        safe_log('before track')
        #bbx = g_tracker.on_track(frame)
        safe_log('after track')
        #safe_log('tracking')
        #g_frame_counter -= 1
        send_bbs('get!')
        te = time.time()
        print(te - ts)
        


if __name__== "__main__": 
    global_init()

    data = np.load(r'C:\Program Files\Teledyne DALSA\Sapera\Demos\Classes\VC\GrabDemo\frame_1.npy')

    recvimg1(data)
    #test()
    
    #t = time.time()
    #for _ in range(1000):
        #mono_to_rgb(data)
    #dt = time.time() - t
    #cv2.imwrite(r'C:\Program Files\Teledyne DALSA\Sapera\Demos\Classes\VC\GrabDemo\a_frame.png', img)
    # cv2.imshow('test', img)
    #print(data.shape)
    
    #g_detector.test_detection()
    #data = np.zeros((612, 480))
    #recvimg1(data)
    # test_call_x(None)
    # test()
    #a = np.array([[1,2,3],[4,5,6],[7,8,9]])
    
    #recvimg(a)