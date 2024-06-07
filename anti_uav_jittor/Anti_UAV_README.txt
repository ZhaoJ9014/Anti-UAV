1. Environment Setup:
Refer to: https://modelscope.cn/models/iic/3rd_Anti-UAV_CVPR23/summary.
note:
a) We recommend that you use python==3.8，or conflicts may occur;
b) For those using NVIDIA RTX 30 & 40 series GPUs，the cuda version 11.8 has been tested successfully;
c) when you run the command
pip install -r requirements/cv.txt
There might be some problems, like unable to find bmt_clipit, clip, panopticapi, videofeatures_clipit. And these errors can be ignored;
d) You need to install jibjpeg4py library and jittor==1.3.8.5.

2. Data Preparation:
Currently, we offer three public datasets for ANTI-UAV task.

ANTI-UAV300:
Google Drive：https://drive.google.com/file/d/1NPYaop35ocVTYWHOYQQHn8YHsM9jmLGr/view
Baidu Drive：https://pan.baidu.com/s/1dJR0VKyLyiXBNB_qfa2ZrA (password: sagx)

ANTI-UAV410:
Baidu Drive：https://pan.baidu.com/s/1PbINXhxc-722NWoO8P2AdQ (password: wfds)

ANTI-UAV600:
https://modelscope.cn/datasets/ly261666/3rd_Anti-UAV/files

Please note that 410 and 600 versions only contain IR videos while 300 version contains both RGB videos and IR videos. In this release, the model is capable of dealing with both RGB data and IR data, so we recommend that you use the 300 version dataset.

3. Running
#Training
Currently, we have some problems with training in Jittor, but you can still try to run the command
python ltr/run_training.py modal modal
in the root path of the project.
Or you can train the model with PyTorch.

Also, if you have any suggestions on how to do it, feel free to open an issue!

#Inference
In the root path of the project, run the command
python pysot_toolkit/test.py

Frequently FAQs (keep updating):
1. Q: When I run the training command, there is an error indicating that setting an array element with a sequence, the requested array has an inhomogeneous shape.
A: In AntiFusion.py, set visible_data = np.array, dtype = object.

