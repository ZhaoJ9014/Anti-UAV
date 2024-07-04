import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

from mmdet.apis import init_detector, inference_detector, show_result_pyplot
import cv2
import numpy as np
import glob
import json



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

        # measure_per_frame.append(self.not_exist(_pred) if not _exist else self.iou(_pred, _gt) if len(_pred) > 1 else 0)
        if not _exist:
            measure_per_frame.append(not_exist(_pred))
        else:

            if len(_gt) < 4 or sum(_gt) == 0:
                continue

            if len(_pred) == 4:
                measure_per_frame.append(iou(_pred, _gt))
            else:
                measure_per_frame.append(0.0)

    return np.mean(measure_per_frame)

def main():

    config_file = 'configs/siamdt_swin_tiny_sgd.py'

    # download the checkpoint from model zoo and put it in `checkpoints/`
    checkpoint_file = 'checkpoints/siamdt_swin_tiny_sgd.pth'
    # checkpoint_file = 'work_dirs/cascade_mask_rcnn_swin_small_patch4_window7_mstrain_480-800_giou_4conv1f_adamw_3x_antiuav/epoch_1.pth'
    device = 'cuda:0'
    # init a detector
    model = init_detector(config_file, checkpoint_file, device=device)
    # inference the demo image


    videofilepath='/data3/publicData/antiUAVtestimages/'

    vis_flag=False

    mode='IR'
    overall_performance = []

    save_results=True

    save_path='results_tiny'

    anno_files = sorted(glob.glob(
                os.path.join(videofilepath, '*/IR_label.json')))

    seq_dirs = [os.path.dirname(f) for f in anno_files]

    seq_names = [os.path.basename(d) for d in seq_dirs]


    for seq_num, seq_name in enumerate(seq_names,start=1):

        print('processing '+seq_name)

        annofile=os.path.join(videofilepath, seq_name+'/IR_label.json')

        with open(annofile, 'r') as f:
            label_res = json.load(f)

        initial_box=label_res['gt_rect'][0]

        output_boxes = []

        img_files=sorted(glob.glob(
                    os.path.join(videofilepath, seq_name+'/img/*.jpg')))

        for index, image_file in enumerate(img_files,start=1):
            
            if index==1:
                
                initial_box=np.array(initial_box,dtype=np.float64)
                output_boxes.append(initial_box.tolist())

                if vis_flag:
                    frame=cv2.imread(image_file)
                    display_name = 'Display: ' + 'AntiUAV'
                    cv2.namedWindow(display_name, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
                    cv2.resizeWindow(display_name, 960, 720)
                    cv2.imshow(display_name, frame)

            # elif index<10:
            else:
                
                result = inference_detector(model, image_file)

                bboxes = np.vstack(result)

                if len(bboxes)>0:
                    box=bboxes[0]
                    output_box=[box[0],box[1],box[2]-box[0],box[3]-box[1]]
                    #if box[4]>0.3:
                    #    output_box=[box[0],box[1],box[2]-box[0],box[3]-box[1]]
                    #else:
                    #    output_box=[]
                else:
                    output_box=[]

                output_box=np.array(output_box,dtype=np.float64)
                output_boxes.append(output_box.tolist())

                if vis_flag:
                    
                    frame=cv2.imread(image_file)

                    if len(output_box)>0:
                    
                        cv2.rectangle(frame, (int(output_box[0]), int(output_box[1])), (int(output_box[2] + output_box[0]), int(output_box[3] + output_box[1])),
                                        (0, 255, 0), 5)

                    font_color = (0, 0, 0)
                    cv2.putText(frame, 'Tracking!', (20, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1,
                                font_color, 1)


                    # Display the resulting frame
                    cv2.imshow(display_name, frame)
                    key = cv2.waitKey(1)

        if vis_flag:
            
            cv2.destroyAllWindows()

        mixed_measure = eval(output_boxes, label_res)

        overall_performance.append(mixed_measure)

        print('%d/%d: %20s %5s Fixed Measure: %.03f' % (seq_num, len(seq_names), seq_name, mode, mixed_measure))


        if save_results:


            if not os.path.exists(save_path):
                os.makedirs(save_path)

            # save result
            output_file = os.path.join(save_path, '%s_%s.txt' % (seq_name, mode))
            with open(output_file, 'w') as f:
                json.dump({'res': output_boxes}, f)

    print('[Overall] %5s Mixed Measure: %.03f\n' % (mode, np.mean(overall_performance)))


if __name__ == '__main__':
    main()
