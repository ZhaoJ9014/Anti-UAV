
from __future__ import absolute_import
import os
import glob
import json
import cv2
import numpy as np
import io

"""
Experiments Setup
"""
# set the dataset path
dataset_path = 'path/to/Anti-UAV410'
# set 'test' or 'val' for evaluation
evaluation_mode = 'test'
# set the path for the predicted tracking results
pred_path = './Tracking_results/Trained_with_antiuav410/SiamDT/'


# mode 1 means the results is formatted by (x,y,w,h)
# mode 2 means the results is formatted by (x1,y1,x2,y2)
mode = 1

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

    if len(pred) == 1 or len(pred) == 0:
        return 1.0
    else:
        return 0.0



def eval(out_res, label_res):

    measure_per_frame = []

    for _pred, _gt, _exist in zip(out_res, label_res['gt_rect'], label_res['exist']):

        if not _exist:
            measure_per_frame.append(not_exist(_pred))
        else:

            if len(_gt)<4 or sum(_gt)==0:
                continue

            if len(_pred)==4:
                measure_per_frame.append(iou(_pred, _gt))
            else:
                measure_per_frame.append(0.0)

            # try:
            #     measure_per_frame.append(iou(_pred, _gt))
            # except:
            #     measure_per_frame.append(0)


        # measure_per_frame.append(not_exist(_pred) if not _exist else iou(_pred, _gt))


    return np.mean(measure_per_frame)




def main():

    evaluation_metrics=['SA Score', 'P Score', 'AUC Score']

    datasetpath = dataset_path + evaluation_mode


    label_files = sorted(glob.glob(
        os.path.join(datasetpath, '*/IR_label.json')))

    video_num = len(label_files)
    overall_performance = []


    if (os.path.exists('eval_details.txt')):
        os.remove('eval_details.txt')

    for video_id, label_file in enumerate(label_files, start=1):

        video_name = os.path.basename(label_file)

        with open(label_file, 'r') as f:
            label_res = json.load(f)

        video_dirs=os.path.dirname(label_file)

        video_dirsbase = os.path.basename(video_dirs)

        pred_file = os.path.join(pred_path, video_dirsbase+'.txt')


        try:
            with open(pred_file, 'r') as f:
                pred_res = json.load(f)
                pred_res=pred_res['res']
        except:
            with open(pred_file, 'r') as f:
                pred_res = np.loadtxt(io.StringIO(f.read().replace(',', ' ')))


        if mode==1:
            pass
        else:
            pred_res[:, 2:] = pred_res[:, 2:] - pred_res[:, :2] + 1

        SA_Score = eval(pred_res, label_res)
        overall_performance.append(SA_Score)

        text = '[%03d/%03d] %25s %15s: %.04f' % (video_id, video_num, video_dirsbase, evaluation_metrics[0], SA_Score)
        with open('eval_details.txt', 'a', encoding='utf-8') as f:
            f.write(text)
            f.write('\n')
        print(text)

    text='[Overall] %25s %15s: %.04f\n' % ('------', evaluation_metrics[0], np.mean(overall_performance))
    with open('eval_details.txt', 'a', encoding='utf-8') as f:
        f.write(text)
    print(text)


if __name__ == '__main__':
    main()
