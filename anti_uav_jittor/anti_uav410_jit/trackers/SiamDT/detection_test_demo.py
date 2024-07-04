import init_paths

from mmdet.apis import init_detector, inference_detector, show_result_pyplot
import cv2

config_file = 'libs/swintransformer/configs/swin/cascade_mask_rcnn_swin_small_patch4_window7_mstrain_480-800_giou_4conv1f_adamw_3x_coco.py'

# download the checkpoint from model zoo and put it in `checkpoints/`
checkpoint_file = 'libs/swintransformer/checkpoints/cascade_mask_rcnn_swin_small_patch4_window7.pth'
device = 'cuda:0'
# init a detector
model = init_detector(config_file, checkpoint_file, device=device)
# inference the demo image

image = 'libs/swintransformer/demo/demo.jpg'

result = inference_detector(model, image)

show_result_pyplot(model, image, result, score_thr=0.3)

# image = model.show_result(image, result, score_thr=0.3)
#
# cv2.imshow('demo', image)
# cv2.waitKey()