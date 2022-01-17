from __future__ import division

import sys

import queue
import pdb
import os
import time
import datetime
import argparse
import struct
import socket
import json
from PIL import Image
import cv2
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import NullLocator
from collections import deque
import numpy as np
from detect_wrapper.Detectoruav import DroneDetection
from tracking_wrapper.dronetracker.trackinguav.evaluation.tracker import Tracker

# sys.path.append((os.path.dirname(__file__)))
sys.path.append(os.path.join(os.path.dirname(__file__),'detect_wrapper'))
sys.path.append(os.path.join(os.path.dirname(__file__),'tracking_wrapper\\dronetracker'))
sys.path.append(os.path.join(os.path.dirname(__file__),'tracking_wrapper\\drtracker'))




## Input video
video_path = os.path.join(os.path.dirname(__file__),'testvideo\\n19.mp4');

magnification = 2

import warnings
warnings.filterwarnings("ignore")

# IP, Port = 0, 0
# IP = "192.168.0.139"
# Port = 9874
# AF_INET, SOCK_DGRAM = socket.AF_INET, socket.SOCK_DGRAM
# udp_socket = socket.socket(AF_INET, SOCK_DGRAM)
# udp_socket.bind(("",9921))

def lefttop2center(bbx):
    obbx=[0,0,bbx[2],bbx[3]]
    obbx[0]=bbx[0]+bbx[2]/2
    obbx[1]=bbx[1]+bbx[3]/2
    return obbx

# class  DroneTracker():
#     def __init__(self):
#         self.tracker = drtracker.CorrFilterTracker(False, True, True)
        
#     def init_track(self, init_box, init_frame):
#         self.tracker.init(init_box, init_frame)
#     def on_track(self, frame):
#         boundingbox = self.tracker.update(frame)
#         boundingbox = [int(x) for x in boundingbox]
#         return boundingbox

def test():
    IRweights_path = os.path.join(os.path.split(os.path.abspath(__file__))[0],'detect_wrapper\\weights\\best.pt');
    # spdb.set_trace()
    time_record=[]
    det_time=[]
    interval=50
    cap = cv2.VideoCapture(video_path)
    
    # vw= VideoWriter("/home/dell/Project_UAV/result/result.avi", fps=30)
    ret, frame = cap.read()
    print(frame.shape)
    # pdb.set_trace()
    oframe = frame.copy()
    drone_det=DroneDetection(IRweights_path=IRweights_path, RGBweights_path=IRweights_path)
    drone_tracker =Tracker()  #DroneTracker()
    first_track=True
    while(ret):
        t1=time.time()
        
        init_box=drone_det.forward_IR(frame)
        t2=time.time()
        det_time.append(t2-t1)
        # if vw is not None:
        #     vw.write(oframe)
        if init_box is not None:
            if first_track:
                drone_tracker.init_track(init_box,frame)
                # first_track=False
            else:
                drone_tracker.change_state(init_box)
            bbx = [int(x) for x in init_box]
            print(bbx)
            
            # cv2.rectangle(oframe,(bbx[0],bbx[1]), (bbx[0]+bbx[2],bbx[1]+bbx[3]), (0,255,0), 2)
            # cv2.imshow("tracking", oframe)
            # cv2.waitKey(1)

            visuframe = cv2.resize(oframe, (oframe.shape[1]*magnification, oframe.shape[0]*magnification), cv2.INTER_LINEAR)
            bbx=[i*magnification for i in bbx]
            cv2.rectangle(oframe,(bbx[0],bbx[1]), (bbx[0]+bbx[2],bbx[1]+bbx[3]), (0,255,0), 2)
            cv2.imshow("tracking", visuframe)
            cv2.waitKey(1)
            
        num=0
        while(num<interval):
            num=num+1
            ret, frame = cap.read()
            

            if ret:
                oframe = frame.copy()
                visuframe = cv2.resize(oframe, (oframe.shape[1]*magnification, oframe.shape[0]*magnification), cv2.INTER_LINEAR)
                t1=time.time()
                outputs=drone_tracker.on_track(frame) 
                t2=time.time()
                time_record.append(t2-t1)
                bbx=[i*magnification for i in outputs]
                print(bbx)

                cv2.rectangle(visuframe,(bbx[0],bbx[1]), (bbx[0]+bbx[2],bbx[1]+bbx[3]), (0,255,0), 2)
                
                cv2.imshow("tracking", visuframe)
                cv2.waitKey(1)
                # if vw is not None:
                #     vw.write(oframe)     
    cap.release()
    # if vw is not None:
    #     vw.release()
    cv2.destroyAllWindows()
    print("done......")
    print('track average time:',np.array(time_record).mean())
    print('detect average time:',np.array(det_time).mean())
    
if __name__=="__main__":
    test()
