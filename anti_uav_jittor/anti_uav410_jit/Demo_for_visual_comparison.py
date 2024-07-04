
from __future__ import absolute_import
import os
import glob
import json
import cv2
import numpy as np
import io


def main(visulization=True):

    dataset_path = 'path/to/Anti-UAV410'

    # test or val
    subset = 'test'

    # show window size
    winwidth=640
    winheight=512

    seq_name = 'all'
    # seq_name = '3700000000002_133828_2'

    save_path='./figures'

    dataset_path = os.path.join(dataset_path, subset)

    # mode 1 means the results is formatted by (x,y,w,h)
    # mode 2 means the results is formatted by (x1,y1,x2,y2)
    trackers=[
        {'name': 'KeepTrack', 'path': './Tracking_results/Trained_with_antiuav410/KeepTrack', 'mode': 1},
        {'name': 'AiATrack', 'path': './Tracking_results/Trained_with_antiuav410/AiATrack/baseline', 'mode': 1},
        {'name': 'SiamBAN', 'path': './Tracking_results/Trained_with_antiuav410/SiamBAN', 'mode': 1},
        {'name': 'SiamCAR', 'path': './Tracking_results/Trained_with_antiuav410/SiamCAR', 'mode': 1},
        {'name': 'Stark', 'path': './Tracking_results/Trained_with_antiuav410/Stark-ST101', 'mode': 1},
        {'name': 'SwinTrack-Tiny', 'path': './Tracking_results/Trained_with_antiuav410/SwinTrack-Tiny', 'mode': 2},
        {'name': 'SiamDT', 'path': './Tracking_results/Trained_with_antiuav410/SiamDT', 'mode': 1},
    ]

    colors = [(255, 0, 0), (0, 0, 255), (0, 255, 255), (255, 0, 255), (0, 97, 255), (255, 255, 0),
             (65, 98, 0)]  # blue, red, yellow, purple, orange, light blue

    label_files = sorted(glob.glob(
        os.path.join(dataset_path, '*/IR_label.json')))

    if visulization:
        cv2.namedWindow("Tracking", 0)
        cv2.resizeWindow("Tracking", winwidth, winheight)

    for video_id, label_file in enumerate(label_files, start=1):

        # groundtruth
        with open(label_file, 'r') as f:
            label_res = json.load(f)

        video_dirs = os.path.dirname(label_file)
        video_dirsbase = os.path.basename(video_dirs)

        if seq_name == 'all':
            pass
        elif video_dirsbase == seq_name:
            pass
        else:
            continue

        image_path = os.path.join(save_path, video_dirsbase)

        if not os.path.exists(image_path):
            os.makedirs(image_path)

        pred_res_total = []

        for trackerid, tracker in enumerate(trackers, start=1):

            pred_file = os.path.join(
                tracker['path'], '%s.txt' % video_dirsbase)

            if tracker['name'] == 'SwinTrack-Tiny':
                pred_file = os.path.join(
                    tracker['path'], 'test_metrics/anti-uav410-%s/%s/bounding_box.txt' % (subset, video_dirsbase))
            try:
                with open(pred_file, 'r') as f:
                    boxes = json.load(f)['res']
            except:
                with open(pred_file, 'r') as f:
                    boxes = np.loadtxt(io.StringIO(f.read().replace(',', ' ')))

            if tracker['mode'] == 2:
                boxes[:, 2:] = boxes[:, 2:] - boxes[:, :2] + 1

            pred_res_total.append(boxes)

        img_files = sorted(glob.glob(
            os.path.join(dataset_path, video_dirsbase, '*.jpg')))

        for frame_id, img_file in enumerate(img_files):

            frame = cv2.imread(img_file)

            _gt = label_res['gt_rect'][frame_id]
            _exist = label_res['exist'][frame_id]
            if _exist:
                cv2.rectangle(frame, (int(_gt[0]), int(_gt[1])), (int(_gt[0] + _gt[2]), int(_gt[1] + _gt[3])),
                              (0, 255, 0))  # (0,255,0) green
            cv2.putText(frame, 'exist' if _exist else 'not exist',
                        (frame.shape[1] // 2 - 20, 30), 1, 2, (0, 255, 0) if _exist else (0, 0, 255), 2)

            for trackerid, (Tbboxes, Tcolor) in enumerate(zip(pred_res_total, colors), start=0):
                bbox = Tbboxes[frame_id]
                cv2.putText(frame, trackers[trackerid]['name'], (20, 60+30*trackerid), 1, 2, Tcolor, 2)
                if len(bbox) > 3:
                    cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3])),
                              Tcolor)

            cv2.putText(frame, video_dirsbase, (frame.shape[1] - 225, frame.shape[0]-10), 1, 1, (255, 255, 0), 2)

            # import pdb;pdb.set_trace()
            if visulization:
                cv2.resizeWindow("Tracking", winwidth, winheight)
                cv2.imshow("Tracking", frame)
                cv2.waitKey(10)

            image_file = os.path.join(image_path, str(frame_id + 1).zfill(4) + '.jpg')
            cv2.imwrite(image_file, frame)

    cv2.destroyAllWindows()




if __name__ == '__main__':
    main(visulization=True)
