#计算检测成功的比例
#在两个阈值下  threshod =0.5 or 0.7

import numpy as np
import random
from typing import List

#[0.5, 0.8, 0.05]
threshod = 0.8
#坐标形式是  x,y,w,h
def calIOU(test_position:List, gt_position:List) -> float:
    xt1 = test_position[0]
    yt1 = test_position[1]
    xb1 = test_position[0] + test_position[2]
    yb1 = test_position[1] + test_position[3]
    xt2 = gt_position[0]
    yt2 = gt_position[1]
    xb2 = gt_position[0] + gt_position[2]
    yb2 = gt_position[1] + gt_position[3]
    #确认交集左上角坐标
    #print(xt1, yt1, xb1, yb1)
    #print(xt2, yt2, xb2, yb2)
    
    xt, yt = max([xt1, xt2]), max([yt1, yt2])
    #确认交集的右下角坐标
    xb, yb = min([xb1, xb2]), min([yb1, yb2])
    #print(xt, yt, xb, yb)
    if(xt > xb or yt > yb):
        return 0.0
    inter = (xb-xt) * (yb-yt)
    #print(xb-xt, yb-yt)
    union = test_position[2] * test_position[3] + gt_position[2] * gt_position[3] - inter
    #print(inter,union)
    iou = inter / union
    #print("\n")
    return iou

def success_count(test : List, gt : List, threshod:float) -> int:
    success_count = 0
    assert len(test) == len(gt)
    for x,y in zip(test, gt):
        iou = calIOU(x, y)  
        if iou  > threshod:
            success_count  += 1
            #print(iou)
    return success_count

def  main(test_path: str, gt_path: str, i:int, thr:float) :
    global sum_percentage
    ftest = open(test_path, 'r')
    fgt = open(gt_path, 'r')
    test_positions = ftest.readlines()
    #构成嵌套列表
    test_positions = [list(map(int,x.split())) for x in test_positions]
    gt_positions = fgt.readlines()  
    gt_positions = [list(map(float,x.split())) for x in gt_positions]
    num = success_count(test_positions, gt_positions, thr)
    percentage = num / len(gt_positions)
    sum_percentage += percentage
    # print(i,' percentage: ', percentage)

if __name__ ==  "__main__":
    sum_percentage = 0
    i=1
    miou=[]
    for thr in [0.5,0.55,0.6,0.65,0.7,0.75,0.8]:
        print("threshlod:",thr)
        sum_percentage = 0
        for i in range(1,46):
            
            main('/home/dell/metric_uav/metric_tracker/pytracking/video_{}_0.txt'.format(i), 
                    '/home/dell/metric_uav/metric_tracker/crop_gt/new_video_{}_0.txt'.format(i), i,thr)

        print("average: ", sum_percentage / 45)
        miou.append(sum_percentage/45)

    print('mIOU:',np.array(miou).mean())
