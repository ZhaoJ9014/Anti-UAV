from lib.test.tracker.basetracker import BaseTracker
import jittor as jt
from lib.train.data.processing_utils import sample_target
# for debug
import cv2
from math import sqrt
import os
from lib.models.uavtrack import build_uavtrack_eh
from lib.test.tracker.tracker_utils import Preprocessor_wo_mask
from lib.utils.box_ops import clip_box, relu_evidence
from lib.test.tracker.tracker_utils import vis_attn_maps

from yolov5.models.common import DetectMultiBackend
from yolov5.utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from yolov5.utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)
from yolov5.utils.augmentations import letterbox
from yolov5.utils.plots import Annotator, colors, save_one_box
from yolov5.utils.torch_utils import select_device, smart_inference_mode
import numpy as np

class MixFormerOnline(BaseTracker):
    def __init__(self, params, dataset_name):
        super(MixFormerOnline, self).__init__(params)
        network = build_uavtrack_eh(params.cfg,  train=False)
        network.load_state_dict(jt.load(self.params.checkpoint, map_location='cpu')['net'], strict=True)
        self.cfg = params.cfg
        self.network = network.cuda()
        self.network.eval()
        self.attn_weights = []

        self.preprocessor = Preprocessor_wo_mask()
        self.state = None

        # for debug
        self.debug = params.debug
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
            self.online_sizes = self.cfg.TEST.ONLINE_SIZES[DATASET_NAME]
            self.online_size = self.online_sizes[0]
        else:
            self.update_intervals = self.cfg.DATA.MAX_SAMPLE_INTERVAL
            self.online_size = 1
        self.update_interval = self.update_intervals[0]
        if hasattr(params, 'online_sizes'):
            self.online_size = params.online_sizes
        print("Online size is: ", self.online_size)
        if hasattr(params, 'update_interval'):
            self.update_interval = params.update_interval
        print("Update interval is: ", self.update_interval)
        if hasattr(params, 'max_score_decay'):
            self.max_score_decay = params.max_score_decay
        else:
            self.max_score_decay = 1.0
        if not hasattr(params, 'vis_attn'):
            self.params.vis_attn = 0
        print("max score decay = {}".format(self.max_score_decay))

    def initialize_yolo(self):
        weights = self.cfg.YOLO.WEIGHTS  # model.pt path(s)
        data = self.cfg.YOLO.DATA # dataset.yaml path
        device = self.cfg.YOLO.DEVICE    # cuda device, i.e. 0 or 0,1,2,3 or cpu

        device = select_device(device)
        model = DetectMultiBackend(weights, device=device, dnn=False, data=data, fp16=False)
        self.yolo_model = model
        return model

    def inilialize_tracker(self, image, target_bbox):
        # forward the template once
        z_patch_arr, _, z_amask_arr = sample_target(image, target_bbox, self.params.template_factor,
                                                    output_sz=self.params.template_size)
        if self.params.vis_attn==1:
            self.z_patch = z_patch_arr
            self.oz_patch = z_patch_arr
        template = self.preprocessor.process(z_patch_arr)
        self.template = template
        self.online_template = template
        if self.online_size > 1:
            with jt.no_grad():
                self.network.set_online(self.template, self.online_template)

        self.online_state = target_bbox

        self.online_image = image
        self.max_pred_score = -1.0
        self.online_max_template = template
        self.online_forget_id = 0

        # save states
        self.state = target_bbox
        self.frame_id = 0
        if self.save_all_boxes:
            '''save all predicted boxes'''
            all_boxes_save = target_bbox * self.cfg.MODEL.NUM_OBJECT_QUERIES
            return {"all_boxes": all_boxes_save}


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
            return {"target_bbox": [0,0,0,0], 'uav_score': 0, 'detection_flag': True}
        else:

            pred_bbox = np.array(det[0][:4].cpu())
            target_bbox = [np.float64(pred_bbox[0]), np.float64(pred_bbox[1]), pred_bbox[2]-pred_bbox[0]+1, pred_bbox[3]-pred_bbox[1]+1]
            self.inilialize_tracker(image, target_bbox)

            return {"target_bbox": target_bbox, 'uav_score': 1, 'detection_flag': False}

    def track(self, image, info: dict = None):
        H, W, _ = image.shape
        self.frame_id += 1
        x_patch_arr, resize_factor, x_amask_arr = sample_target(image, self.state, self.params.search_factor,
                                                                output_sz=self.params.search_size)  # (x1, y1, w, h)
        search = self.preprocessor.process(x_patch_arr)
        with jt.no_grad():
            if self.online_size==1:
                # for visualize attention maps
                if self.params.vis_attn==1 and self.frame_id % 200 == 0:
                    attn_weights = []
                    hooks = []
                    for i in range(len(self.network.backbone.stage2.blocks)):
                        hooks.append(self.network.backbone.stage2.blocks[i].attn.attn_drop.register_forward_hook(
                            lambda self, input, output: attn_weights.append(output)))
                out_dict, _ = self.network(self.template, self.online_template, search, run_score_head=True)
                if self.params.vis_attn==1 and self.frame_id % 200 == 0:
                    for hook in hooks:
                        hook.remove()
                    # attn0(t_ot) / 1(t_ot) / 2(t_ot_s)
                    # shape: torch.Size([1, 6, 64, 32]), torch.Size([1, 6, 64, 32]), torch.Size([1, 6, 400, 132])
                    # vis attn weights: online_template-to-template
                    vis_attn_maps(attn_weights[::3], q_w=8, k_w=4, skip_len=16, x1=self.oz_patch, x2=self.z_patch,
                                  x1_title='Online Template', x2_title='Template',
                                  save_path= 'vis_attn_weights/t2ot_vis/%04d' % self.frame_id)
                    # vis attn weights: template-to-online_template
                    vis_attn_maps(attn_weights[1::3], q_w=8, k_w=4, skip_len=0, x1=self.z_patch, x2=self.oz_patch,
                                  x1_title='Template', x2_title='Online Template',
                                  save_path='vis_attn_weights/ot2t_vis/%04d' % self.frame_id)
                    # vis attn weights: template-to-search
                    vis_attn_maps(attn_weights[2::3], q_w=20, k_w=4, skip_len=0, x1=self.z_patch, x2=x_patch_arr,
                                  x1_title='Template', x2_title='Search',
                                  save_path='vis_attn_weights/s2t_vis/%04d' % self.frame_id)
                    # vis attn weights: online_template-to-search
                    vis_attn_maps(attn_weights[2::3], q_w=20, k_w=4, skip_len=16, x1=self.oz_patch, x2=x_patch_arr,
                                  x1_title='Online Template', x2_title='Search',
                                  save_path='vis_attn_weights/s2ot_vis/%04d' % self.frame_id)
                    # vis attn weights: search-to-search
                    vis_attn_maps(attn_weights[2::3], q_w=20, k_w=10, skip_len=32, x1=x_patch_arr, x2=x_patch_arr,
                                  x1_title='Search1', x2_title='Search2', idxs=[(160, 160)],
                                  save_path='vis_attn_weights/s2s_vis/%04d' % self.frame_id)
                    print("save vis_attn of frame-{} done.".format(self.frame_id))
            else:
                out_dict, _ = self.network.forward_test(search, run_score_head=True)
        pred_boxes = out_dict['pred_boxes'].view(-1, 4)

        # pred_score = out_dict['pred_scores'].view(1).sigmoid().item()
        evidence = relu_evidence(out_dict['pred_scores'].squeeze(1))
        alpha = evidence + 1
        uncertainty = 2 / jt.sum(alpha, dim=1, keepdim=True)  # uncertainty
        prob = alpha / jt.sum(alpha, dim=1, keepdim=True) # the score
        _, preds = jt.max(out_dict['pred_scores'].squeeze(1), 1) # class:target or background
        # Baseline: Take the mean of all pred boxes as the final result
        pred_box = (pred_boxes.mean(dim=0) * self.params.search_size / resize_factor).tolist()  # (cx, cy, w, h) [0,1]
        # get the final box result
        self.state = clip_box(self.map_box_back(pred_box, resize_factor), H, W, margin=5)


        if preds == 0 or (preds == 1 and uncertainty > self.params.evidential_threshold):
            out = self.initialize(image, self.yolo_model)
            return out


        self.max_pred_score = self.max_pred_score * self.max_score_decay
        # update template
        # if prob[0][1] > 0.5 and prob[0][1] > self.max_pred_score:
        #     z_patch_arr, _, z_amask_arr = sample_target(image, self.state,
        #                                                 self.params.template_factor,
        #                                                 output_sz=self.params.template_size)  # (x1, y1, w, h)
        #     self.online_max_template = self.preprocessor.process(z_patch_arr)
        #     if self.params.vis_attn == 1:
        #         self.oz_patch_max = z_patch_arr
        #     self.max_pred_score = prob[0][1]
        # if self.frame_id % self.update_interval == 0:
        #     if self.online_size == 1:
        #         self.online_template = self.online_max_template
        #         if self.params.vis_attn == 1:
        #             self.oz_patch = self.oz_patch_max
        #     elif self.online_template.shape[0] < self.online_size:
        #         self.online_template = torch.cat([self.online_template, self.online_max_template])
        #     else:
        #         self.online_template[self.online_forget_id:self.online_forget_id+1] = self.online_max_template
        #         self.online_forget_id = (self.online_forget_id + 1) % self.online_size
        #
        #     if self.online_size > 1:
        #         with torch.no_grad():
        #             self.network.set_online(self.template, self.online_template)
        #
        #     self.max_pred_score = -1
        #     self.online_max_template = self.template

        # for debug
        if self.debug:
            x1, y1, w, h = self.state
            # image_BGR = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            cv2.rectangle(image, (int(x1),int(y1)), (int(x1+w),int(y1+h)), color=(0,0,255), thickness=2)
            save_path = os.path.join(self.save_dir, "%04d.jpg" % self.frame_id)
            cv2.imwrite(save_path, image)
        if self.save_all_boxes:
            '''save all predictions'''
            all_boxes = self.map_box_back_batch(pred_boxes * self.params.search_size / resize_factor, resize_factor)
            all_boxes_save = all_boxes.view(-1).tolist()  # (4N, )
            return {"target_bbox": self.state,
                    "all_boxes": all_boxes_save}
        else:
            return {"target_bbox": self.state, 'uav_score': prob[0][1], 'detection_flag': False}

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

    def get_target(self, im, target_bb, output_sz=16):
        if not isinstance(target_bb, list):
            x, y, w, h = target_bb.tolist()
        else:
            x, y, w, h = target_bb
        x1 = int(x)
        x2 = int(x1 + w)
        y1 = int(y)
        y2 = int(y1 + h)
        # Crop target
        im_crop = im[y1 : y2, x1 : x2, :]
        target_patch = cv2.resize(im_crop, (output_sz, output_sz))
        return target_patch

    def calculate_similarity(self, v1, v2):
        def dot_product(v1, v2):

            return jt.sum(v1*v2)

        def magnitude(vector):
            """Returns the numerical length / magnitude of the vector."""
            return jt.sqrt(dot_product(vector, vector))

        return dot_product(v1, v2) / (magnitude(v1) * magnitude(v2) + .00000000001)


def get_tracker_class():
    return MixFormerOnline