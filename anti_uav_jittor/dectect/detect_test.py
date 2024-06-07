from dectect.dect_predict import FRCNN
from PIL import Image
frcnn = FRCNN()

import os
# video_path = "D:\study\\track\dataset\Anti-UAV-RGBT\\test\\20190925_124000_1_4\\infrared.mp4"
# video_save_path = "result/result.avi"
# video_fps = 25.0

img_path = 'D:\study\\track\dataset\Anti-UAV-RGBT\\train\\20190925_131530_1_2_occ\\infrared\\infraredI0000.jpg'

image = Image.open(img_path)

top_conf, top_boxes = frcnn.detect_image(image, crop = False, count = False)
print(top_boxes)