from lib.test.tracker.basetracker import BaseTracker
import jittor as jt
from lib.train.data.processing_utils import sample_target
# for debug
import cv2
import os
from lib.models.uavtrack import build_uavtrack
from lib.test.tracker.tracker_utils import Preprocessor_wo_mask
from lib.utils.box_ops import clip_box

from yolov5.models.common import DetectMultiBackend
from yolov5.utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from yolov5.utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)
from yolov5.utils.augmentations import letterbox
from yolov5.utils.plots import Annotator, colors, save_one_box
from yolov5.utils.torch_utils import select_device, smart_inference_mode
import numpy as np


class UAVTrack(BaseTracker):
    def __init__(self, params, dataset_name):
        super(UAVTrack, self).__init__(params)
        network = build_uavtrack(params.cfg)
        network.load_state_dict(jt.load(self.params.checkpoint, map_location='cpu')['net'], strict=True)
        self.cfg = params.cfg
        self.network = network.cuda()
        self.network.eval()
        self.preprocessor = Preprocessor_wo_mask()
        self.state = None
        # for debug
        self.debug = False
        self.frame_id = 0
        if self.debug:
            self.save_dir = "debug"
            if not os.path.exists(self.save_dir):
                os.makedirs(self.save_dir)
        # for save boxes from all queries
        self.save_all_boxes = params.save_all_boxes
        # self.z_dict1 = {}

        # Set the update interval
        DATASET_NAME = dataset_name.upper()
        if hasattr(self.cfg.TEST.UPDATE_INTERVALS, DATASET_NAME):
            self.update_intervals = self.cfg.TEST.UPDATE_INTERVALS[DATASET_NAME]
        else:
            self.update_intervals = [self.cfg.DATA.MAX_SAMPLE_INTERVAL]
        print("Update interval is: ", self.update_intervals)

    def initialize_yolo(self):
        weights = self.cfg.YOLO.WEIGHTS  # model.pt path(s)
        data = self.cfg.YOLO.DATA # dataset.yaml path
        device = self.cfg.YOLO.DEVICE    # cuda device, i.e. 0 or 0,1,2,3 or cpu
        device = select_device(device)
        model = DetectMultiBackend(weights, device=device, dnn=False, data=data, fp16=False)

        return model

    def initialize(self, image, model):

        # detect the uav target by yolov5s
        imgsz = self.cfg.YOLO.IMGSZ  # inference size (height, width)
        conf_thres = self.cfg.YOLO.CONF_THRES # 0.001 #0.25  # confidence threshold
        iou_thres = self.cfg.YOLO.IOU_THRES # 0.6 #0.45  # NMS IOU threshold
        max_det = self.cfg.YOLO.MAX_DET  # maximum detections per image
        augment = self.cfg.YOLO.AUGMENT  # augmented inference
        classes = self.cfg.YOLO.CLASSES  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms = self.cfg.YOLO.AGNOSTIC_NMS # class-agnostic NMS
        hide_labels = self.cfg.YOLO.HIDE_LABELS # hide labels
        hide_conf = self.cfg.YOLO.HIDE_CONF # hide confidences
        weights = self.cfg.YOLO.WEIGHTS  # model.pt path(s)
        data = self.cfg.YOLO.DATA # dataset.yaml path
        device = self.cfg.YOLO.DEVICE    # cuda device, i.e. 0 or 0,1,2,3 or cpu

        device = select_device(device)
        # model = DetectMultiBackend(weights, device=device, dnn=False, data=data, fp16=False)
        stride, names, pt = model.stride, model.names, model.pt
        imgsz = check_img_size(imgsz, s=stride)  # check image size

        # resize_im
        im = letterbox(image, imgsz, stride=stride, auto=True)[0]  # padded resize
        im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        im = np.ascontiguousarray(im)  # contiguous
        # Run inference
        bs = 1
        model.warmup(imgsz=(1 if pt else bs, 3, *imgsz))  # warmup
        seen, windows, dt = 0, [], (Profile(), Profile(), Profile())
        with dt[0]:
            im = jt.Var(im)
            im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]
        # Inference
        with dt[1]:
            visualize = False
            pred = model(im, augment=augment, visualize=visualize)
        # NMS
        with dt[2]:
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

        # Process predictions
        for i, det in enumerate(pred):  # per image
            seen += 1
            im0 = image.copy()
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

        if len(det) == 0:
            return {"target_bbox": [0,0,0,0]}
        else:
            pred_bbox = np.array(det[0][:4].cpu())
            target_bbox = [np.float64(pred_bbox[0]), np.float64(pred_bbox[1]), pred_bbox[2]-pred_bbox[0]+1, pred_bbox[3]-pred_bbox[1]+1]
            # forward the template once
            z_patch_arr, _, z_amask_arr = sample_target(image, target_bbox, self.params.template_factor,
                                                        output_sz=self.params.template_size)
            template = self.preprocessor.process(z_patch_arr)
            self.template = template

            self.online_template = template
            # print("template shape: {}".format(template.shape))
            # with torch.no_grad():
            #     self.z_dict1 = self.network.forward_backbone(template)
            # save states
            self.state = target_bbox
            self.frame_id = 0
            if self.save_all_boxes:
                '''save all predicted boxes'''
                all_boxes_save = target_bbox * self.cfg.MODEL.NUM_OBJECT_QUERIES
                return {"all_boxes": all_boxes_save}
            return {"target_bbox": target_bbox}

    def track(self, image, info: dict = None):
        H, W, _ = image.shape
        self.frame_id += 1
        # print("frame id: {}".format(self.frame_id))
        x_patch_arr, resize_factor, x_amask_arr = sample_target(image, self.state, self.params.search_factor,
                                                                output_sz=self.params.search_size)  # (x1, y1, w, h)
        search = self.preprocessor.process(x_patch_arr)
        # print("search shape: {}".format(search.shape))
        with jt.no_grad():
            # x_dict = self.network.forward_backbone(search)
            # # merge the template and the search
            # feat_dict_list = [self.z_dict1, x_dict]
            # seq_dict = merge_template_search(feat_dict_list)
            # # run the transformer
            # out_dict, _, _ = self.network.forward_transformer(seq_dict=seq_dict, run_box_head=True)
            out_dict, _ = self.network(self.template, self.online_template, search)
            # print("out_dict: {}".format(out_dict))

        pred_boxes = out_dict['pred_boxes'].view(-1, 4)
        # Baseline: Take the mean of all pred boxes as the final result
        pred_box = (pred_boxes.mean(dim=0) * self.params.search_size / resize_factor).tolist()  # (cx, cy, w, h) [0,1]
        # get the final box result
        self.state = clip_box(self.map_box_back(pred_box, resize_factor), H, W, margin=10)

        # update template
        for idx, update_i in enumerate(self.update_intervals):
            if self.frame_id % update_i == 0:
                z_patch_arr, _, z_amask_arr = sample_target(image, self.state, self.params.template_factor,
                                                            output_sz=self.params.template_size)  # (x1, y1, w, h)
                self.online_template = self.preprocessor.process(z_patch_arr)


        # for debug
        if self.debug:
            x1, y1, w, h = self.state
            image_BGR = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            cv2.rectangle(image_BGR, (int(x1),int(y1)), (int(x1+w),int(y1+h)), color=(0,0,255), thickness=2)
            save_path = os.path.join(self.save_dir, "%04d.jpg" % self.frame_id)
            cv2.imwrite(save_path, image_BGR)
        if self.save_all_boxes:
            '''save all predictions'''
            all_boxes = self.map_box_back_batch(pred_boxes * self.params.search_size / resize_factor, resize_factor)
            all_boxes_save = all_boxes.view(-1).tolist()  # (4N, )
            return {"target_bbox": self.state,
                    "all_boxes": all_boxes_save}
        else:
            return {"target_bbox": self.state}

    def map_box_back(self, pred_box: list, resize_factor: float):
        cx_prev, cy_prev = self.state[0] + 0.5 * self.state[2], self.state[1] + 0.5 * self.state[3]
        cx, cy, w, h = pred_box
        half_side = 0.5 * self.params.search_size / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return [cx_real - 0.5 * w, cy_real - 0.5 * h, w, h]

    def map_box_back_batch(self, pred_box: jt.Var, resize_factor: float):
        cx_prev, cy_prev = self.state[0] + 0.5 * self.state[2], self.state[1] + 0.5 * self.state[3]
        cx, cy, w, h = pred_box.unbind(-1) # (N,4) --> (N,)
        half_side = 0.5 * self.params.search_size / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return jt.stack([cx_real - 0.5 * w, cy_real - 0.5 * h, w, h], dim=-1)


def get_tracker_class():
    return UAVTrack
