# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import os
import sys
env_path = os.path.join(os.path.dirname(__file__), '..')
print(env_path)
if env_path not in sys.path:
    sys.path.append(env_path)
import argparse
import os
import time
import cv2
import numpy as np
from pysot_toolkit.toolkit.datasets.Fusiondataset import AntiFusion
from pysot_toolkit.bbox import get_axis_aligned_bbox
from pysot_toolkit.toolkit.datasets import DatasetFactory

from pysot_toolkit.trackers.tracker import Tracker
from pysot_toolkit.trackers.net_wrappers import NetWithBackbone

net_path = '/data01/xjy/code/anti_cp/model_1/Modal_FPN_ep0044.pth'

def iou(bbox1, bbox2):
    """
    Calculates the intersection-over-union of two bounding boxes.
    Args:
        bbox1 (numpy.array, list of floats): bounding box in format x,y,w,h.
        bbox2 (numpy.array, list of floats): bounding box in format x,y,w,h.
    Returns:
        int: intersection-over-onion of bbox1, bbox2
    """
    bbox1 = [float(x) for x in bbox1]
    bbox2 = [float(x) for x in bbox2]

    (x0_1, y0_1, w1_1, h1_1) = bbox1
    (x0_2, y0_2, w1_2, h1_2) = bbox2
    x1_1 = x0_1 + w1_1
    x1_2 = x0_2 + w1_2
    y1_1 = y0_1 + h1_1
    y1_2 = y0_2 + h1_2
    # get the overlap rectangle
    overlap_x0 = max(x0_1, x0_2)
    overlap_y0 = max(y0_1, y0_2)
    overlap_x1 = min(x1_1, x1_2)
    overlap_y1 = min(y1_1, y1_2)

    # check if there is an overlap
    if overlap_x1 - overlap_x0 <= 0 or overlap_y1 - overlap_y0 <= 0:
        return 0

    # if yes, calculate the ratio of the overlap to each ROI size and the unified size
    size_1 = (x1_1 - x0_1) * (y1_1 - y0_1)
    size_2 = (x1_2 - x0_2) * (y1_2 - y0_2)
    size_intersection = (overlap_x1 - overlap_x0) * (overlap_y1 - overlap_y0)
    size_union = size_1 + size_2 - size_intersection

    return size_intersection / size_union

def not_exist(pred):
    return (len(pred) == 1 and pred[0] == 0) or len(pred) == 0


def convert_lusion(original_image,size,ann):


    # 假设标注信息是矩形框的坐标 (x, y, width, height)
    original_annotation = ann

    # 获取原始图像的大小
    original_height, original_width, _ = original_image.shape

    # 更改图像大小为640x512
    new_size = size
    resized_image = cv2.resize(original_image, new_size)

    # 计算缩放比例
    scale_x = new_size[0] / original_width
    scale_y = new_size[1] / original_height

    # 缩放标注信息
    if sum(original_annotation) == 0:
        original_annotation = [0, 0, 0, 0]
    new_x = int(original_annotation[0] * scale_x)
    new_y = int(original_annotation[1] * scale_y)
    new_width = int(original_annotation[2] * scale_x)
    new_height = int(original_annotation[3] * scale_y)

    resized_annotation = (new_x, new_y, new_width, new_height)

    return resized_image, resized_annotation


def eval(out_res, label_res):
    measure_per_frame = []
    for _pred, _gt, _exist in zip(out_res, label_res['gt_rect'], label_res['exist']):
        measure_per_frame.append(not_exist(_pred) if not _exist else iou(_pred, _gt) if len(_pred) > 1 else 0)
    return np.mean(measure_per_frame)

def _record(record_file, boxes, times):
    # record bounding boxes
    record_dir = os.path.dirname(record_file)
    if not os.path.isdir(record_dir):
        os.makedirs(record_dir)
    np.savetxt(record_file, boxes, fmt='%.3f', delimiter=',')
    while not os.path.exists(record_file):
        print('warning: recording failed, retrying...')
        np.savetxt(record_file, boxes, fmt='%.3f', delimiter=',')
    print('  Results recorded at', record_file)

    # record running times
    time_dir = os.path.join(record_dir, 'times')
    if not os.path.isdir(time_dir):
        os.makedirs(time_dir)
    time_file = os.path.join(time_dir, os.path.basename(
        record_file).replace('.txt', '_time.txt'))
    np.savetxt(time_file, times, fmt='%.8f')

def img_show(image,gt_bbox,pred_bbox):
    img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    # size = img.shape
    gt_bbox = list(map(int, gt_bbox))
    if sum(gt_bbox) == 0:
        gt_bbox = [0,0,0,0]


    pred_bbox = list(map(int, pred_bbox))
    cv2.rectangle(img, (gt_bbox[0], gt_bbox[1]),
                  (gt_bbox[0] + gt_bbox[2], gt_bbox[1] + gt_bbox[3]), (0, 255, 0), 3)
    cv2.rectangle(img, (pred_bbox[0], pred_bbox[1]),
                  (pred_bbox[0] + pred_bbox[2], pred_bbox[1] + pred_bbox[3]), (0, 255, 255), 3)

    cv2.putText(img, 'frame' +str(idx), (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    img = cv2.resize(img,(640,512))


    return img

# create model
net = NetWithBackbone(net_path=net_path, use_gpu=True)
net.load_network()
test_dataset = AntiFusion(1)
scores = {}
tracker = Tracker(name='modals', net=net, window_penalty=0.3, exemplar_size=128, instance_size=256)
results_file = open('result_test_2.txt', 'w+', encoding='utf-8')
acc_videos_ir = []
acc_videos_rgb = []

for v_idx in range(len(test_dataset)):
    videos = test_dataset[v_idx]
    rgb_video, ir_video = videos[0], videos[1]
    # if rgb_video.video_id != '20190926_195921_1_6':
    #     continue
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    videoWriter = cv2.VideoWriter('test{}_or.avi'.format(rgb_video.video_id),
                                  fourcc, 30, (640*2,512))
    toc = 0
    retemplate = False
    track_times = []
    video_acc_rgb = []
    video_acc_ir = []
    boxes_rgb = np.zeros((len(rgb_video), 4))
    boxes_ir = np.zeros((len(rgb_video), 4))
    times = np.zeros(len(rgb_video))
    for idx in range(len(rgb_video)):
        img_ir,  gt_bbox_ir = ir_video[idx]
        img_rgb, gt_bbox_rgb = rgb_video[idx]
        start_time = time.time()
        tic = cv2.getTickCount()
        if idx == 0 or retemplate == True:
            if sum(gt_bbox_ir) == 0 or sum(gt_bbox_rgb) == 0 :
                retemplate = True
                continue
            init_info_ir = {'init_bbox': gt_bbox_ir}
            boxes_ir[idx] = gt_bbox_ir
            # img_rgb,gt_bbox_rgb = convert_lusion(img_rgb,(640,512),gt_bbox_rgb)

            init_info_rgb = {'init_bbox': gt_bbox_rgb}
            print(gt_bbox_rgb)
            img = {'ir':img_ir, 'rgb':img_rgb}
            boxes_rgb[idx] = gt_bbox_rgb
            tracker.multi_ininize(images=img, rgb_info=init_info_rgb,ir_info=init_info_ir)
            retemplate = False
        else:
            # img_rgb, gt_bbox_rgb = convert_lusion(img_rgb, (640, 512), gt_bbox_rgb)
            img = {'ir': img_ir, 'rgb': img_rgb}

            outputs = tracker.track_only_local(img)
            ir_out,rgb_out = outputs['ir'],outputs['rgb']
            img_rect_rgb = img_show(img_rgb,gt_bbox_rgb,rgb_out)
            img_rect_ir = img_show(img_ir, gt_bbox_ir, ir_out)
            boxes_rgb[idx] = rgb_out
            boxes_ir[idx] = ir_out
            new_img = np.hstack([img_rect_rgb, img_rect_ir])
            videoWriter.write(new_img)
            # cv2.imshow('test', new_img)
            # cv2.waitKey(1)
            cv2.imwrite('/data01/xjy/code/anti_cp/data/inference_results/'+'1.png', new_img)
            print("inference at pic " + str(idx))
            if sum(ir_out) == 0 :
                if len(gt_bbox_ir) == 0:
                    acc_videos_ir.append(1)
                else:
                    acc_videos_ir.append(0)
            else:
                if len(gt_bbox_ir) == 0:
                    acc_videos_ir.append(0)
                else:
                    acc_videos_ir.append(iou(ir_out,gt_bbox_ir))
            if sum(rgb_out) == 0 :
                if len(gt_bbox_rgb) == 0:
                    acc_videos_rgb.append(1)
                else:
                    acc_videos_rgb.append(0)
            else:
                if len(gt_bbox_rgb) == 0:
                    acc_videos_rgb.append(0)
                else:
                    acc_videos_rgb.append(iou(rgb_out,gt_bbox_rgb))
            # print('ir',acc_videos_ir[-1])
            video_acc_ir.append((acc_videos_ir[-1]))
            video_acc_rgb.append(acc_videos_rgb[-1])
        times[idx] = time.time() - start_time
    record_file_ir = os.path.join(
        'result', 'UNFPN_IR', '%s.txt' % rgb_video.video_id)
    record_file_rgb = os.path.join(
        'result', 'UNFPN_RGB', '%s.txt' % rgb_video.video_id)
    _record(record_file_ir,boxes_ir,times)
    _record(record_file_rgb,boxes_rgb,times)
    # results_file.write(str(np.mean(video_acc_ir))+'\t'+str(np.mean(video_acc_rgb)))
    # results_file.write('\n')
    videoWriter.release()
    print(np.mean(video_acc_ir),np.mean(video_acc_rgb),'\t'+rgb_video.video_id)
acc_ir = np.mean(acc_videos_ir)
acc_rgb = np.mean(acc_videos_rgb)
acc = (acc_ir + acc_rgb)/2
results_file.write('ir'+str(np.mean(acc_ir))+'\n')
results_file.write('rgb'+str(np.mean(acc_rgb))+'\n')
results_file.write('acc_total'+str(np.mean(acc))+'\n')
print('ir'+str(np.mean(acc_ir)))
print('rgb'+str(np.mean(acc_rgb)))
print('acc_total'+str(np.mean(acc)))
results_file.close()
