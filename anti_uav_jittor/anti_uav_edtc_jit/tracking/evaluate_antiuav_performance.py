import os
import glob
import json
import cv2
import numpy as np
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
    return (pred[0] == 0 and pred[2]==0) or len(pred) == 0

def eval(out_res, label_res):
    measure_per_frame = []
    penalty_measures = []
    for _pred, _gt, _exist in zip(out_res, label_res['gt_rect'], label_res['exist']):
        measure_per_frame.append(not_exist(_pred) if not _exist else iou(_pred, _gt) if len(_pred) > 1 else 0)
        if _exist:
            if (len(_pred) > 1 and iou(_pred, _gt) < 1e-5) or not_exist(_pred):
                penalty_measures.append(1)
            else:
                penalty_measures.append(0)
        else:
            penalty_measures.append(0)

    return np.mean(measure_per_frame) - 0.2 * np.mean(penalty_measures) ** 0.3

dataset_root = '/path/to/test set'
list_file = os.path.join(dataset_root, 'list.txt')
with open(list_file, 'r') as f:
    video_files = [video.split('\n')[0] for video in f.readlines()]
results_root = 'path/to/results'
overall_performance = []
video_id = 0
video_num = len(video_files)

for video in video_files:
    video_id += 1
    # load groundtruth
    video_path = os.path.join(dataset_root, video)
    label_file = os.path.join(video_path, 'IR_label.json')
    with open(label_file, 'r') as f:
        label_res = json.load(f)
    print(video)

    # load predicted results
    res_file = os.path.join(results_root, '%s.txt' % video)
    with open(res_file, 'r') as f:
        bboxes_list = [i.split('\n')[0].split('\t') for i in f.readlines()]
        bboxes = [(float(i[0]), float(i[1]), float(i[2]), float(i[3])) for i in bboxes_list]
    out_res = bboxes
    mixed_measure = eval(out_res, label_res)
    overall_performance.append(mixed_measure)
    print('[%03d/%03d] %20s Fixed Measure: %.03f' % (video_id, video_num, video, mixed_measure))

print('[Overall] Mixed Measure: %.03f\n' % np.mean(overall_performance))
