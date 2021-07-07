import os
import shutil
import json
from typing import Dict, List
import torch
import math
import random
from utils.general import box_iou
import sys

#sys.path.append("/home/dell/Project_UAV/detect_wrapper")
ground_truth_imgs_path = '/home/dell/metric_uav/metric_detector/drone_select/images/val'
ground_truth_labels_path = '/home/dell/metric_uav/metric_detector/drone_select/labels/val'

test_result_labels_path = '/home/dell/metric_uav/metric_detector/runs/pred/autolabels'


def read_labels(path: str, is_gen: bool = False, width: int = 640, height: int = 512) -> dict:
    ret = {}
    offset = 0#1 if is_gen else 0
    for gd_file_name in os.listdir(path):
        file_path = os.path.join(path, gd_file_name)
        with open(file_path, 'r') as f:
            lines = f.readlines()
            name = os.path.splitext(gd_file_name)[0]
            item = []
            if not is_gen:
                assert len(lines) == 1
            if len(lines) > 0:
                for line in lines:
                    words = line.strip().split()
                    x1 = round(float(words[offset + 1]) * width)
                    y1 = round(float(words[offset + 2]) * height)
                    w = round(float(words[offset + 3]) * width)
                    h = round(float(words[offset + 4]) * height)
                    # if (not is_gen) and (w < 15 or h < 15):
                    #     continue
                    x2 = x1 + w
                    y2 = y1 + h
                    item.append([x1, y1, x2, y2])
                ret[name] = torch.as_tensor(item, dtype=torch.float64)
            else:
                ret[name] = torch.zeros(size=(0, 4), dtype=torch.float64)
    return ret


target_num = 10000

iou_thres = 0.5

if __name__ == '__main__':

    truth = read_labels(ground_truth_labels_path)
    pred = read_labels(test_result_labels_path, is_gen=True)

    all_num = len(truth)

    for k in truth:
        if k not in pred:
            pred[k] = torch.zeros(size=(0, 4), dtype=torch.float64)

    assert set(truth.keys()) == set(pred.keys())

    tp_all, fp_all, tn_all, fn_all = 0, 0, 0, 0
    tp_img, fp_img, tn_img, fn_img = 0, 0, 0, 0

    pred_measurement = dict()

    for k in truth.keys():
        t = truth[k]
        p = pred[k]

        iou_all: torch.Tensor = box_iou(t, p)
        match_all: torch.Tensor = iou_all >= iou_thres

        m, n = iou_all.shape

        assert m == 1

        tp, fp, tn, fn = 0, 0, 0, 0

        for i in range(n):
            if match_all[0, i]:
                tp += 1
                matched = True
                break

        fp = n - tp
        fn = 1 - tp

        tp_all += tp
        fp_all += fp
        tn_all += tn
        fn_all += fn

        pred_measurement[k] = {
            'tp': tp,
            'fp': fp,
            'tn': tn,
            'fn': fn,
        }

    # print('tp =', tp_all)
    # print('tn =', tn_all)
    # print('fp =', fp_all)
    # print('fn =', fn_all)

    classies = {
        'tp': 0,
        'tp+fp': 0,
        'fn+fp': 0,
        'fn': 0,
    }

    classies_keys = {
        'tp': set(),
        'tp+fp': set(),
        'fn+fp': set(),
        'fn': set(),
    }

    for (k, v) in pred_measurement.items():
        tp, fp, tn, fn = v['tp'], v['fp'], v['tn'], v['fn']
        if tp == 1 and fp == 0:
            classies['tp'] += 1
            classies_keys['tp'].add(k)
        elif tp == 1 and fp > 0:
            classies['tp+fp'] += 1
            classies_keys['tp+fp'].add(k)
        elif fn == 1 and fp > 0:
            classies['fn+fp'] += 1
            classies_keys['fn+fp'].add(k)
        elif fn == 1 and fp == 0:
            classies['fn'] += 1
            classies_keys['fn'].add(k)
        else:
            assert False

    print(classies)

    print('Accuracy:',classies['tp']/10000)
