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
sys.path.append("/home/dell/Project_UAV/tracking_wrapper/dronetracker")
sys.path.append("/home/dell/Project_UAV/tracking_wrapper/drtracker")
from trackinguav.evaluation.tracker import Tracker
import drtracker
from visualization import VideoWriter
sys.path.append("/home/dell/Project_UAV/detect_wrapper")
from collections import deque
import numpy as np
from Detectoruav import DroneDetection

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

class  DroneTracker():
    def __init__(self):
        self.tracker = drtracker.CorrFilterTracker(False, True, True)
        
    def init_track(self, init_box, init_frame):
        self.tracker.init(init_box, init_frame)
    def on_track(self, frame):
        boundingbox = self.tracker.update(frame)
        boundingbox = [int(x) for x in boundingbox]
        return boundingbox

def test():
    time_record=[]
    det_time=[]
    interval=300
    cap = cv2.VideoCapture('/home/dell/Project_UAV/testvideo/n8.mp4')
    
    vw= VideoWriter("/home/dell/Project_UAV/result/result.avi", fps=30)
    ret, frame = cap.read()
    print(frame.shape)

    
    oframe = frame.copy()
    drone_det=DroneDetection()
    drone_tracker =Tracker()  #DroneTracker()
    first_track=True
    while(ret):
        t1=time.time()
        init_box=drone_det.forward_IR(frame)
        t2=time.time()
        det_time.append(t2-t1)
        if vw is not None:
            vw.write(oframe)
        if init_box is not None:
            if first_track:
                drone_tracker.init_track(init_box,frame)
                first_track=False
            else:
                drone_tracker.change_state(init_box)
            bbx = [int(x) for x in init_box]
            print(bbx)
            cv2.rectangle(oframe,(bbx[0],bbx[1]), (bbx[0]+bbx[2],bbx[1]+bbx[3]), (0,255,0), 2)
        
            cv2.imshow("tracking", oframe)
            cv2.waitKey(1)
            
        num=0
        while(num<interval):
            num=num+1
            ret, frame = cap.read()
            

            if ret:
                oframe = frame.copy()
                t1=time.time()
                outputs=drone_tracker.on_track(frame) 
                t2=time.time()
                time_record.append(t2-t1)
                bbx=outputs
                print(bbx)

                cv2.rectangle(oframe,(bbx[0],bbx[1]), (bbx[0]+bbx[2],bbx[1]+bbx[3]), (0,255,0), 2)
                
                cv2.imshow("tracking", oframe)
                cv2.waitKey(1)
                if vw is not None:
                    vw.write(oframe)     
    cap.release()
    if vw is not None:
        vw.release()
    cv2.destroyAllWindows()
    print("done......")
    print('track average time:',np.array(time_record).mean())
    print('detect average time:',np.array(det_time).mean())
    
if __name__=="__main__":
    test()
