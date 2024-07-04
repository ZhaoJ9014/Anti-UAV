# Anti-Unmanned Aerial Vehicle (UAV)

## License

The project of Anti-UAV is released under the MIT License.


****
## News
:white_check_mark: **`07 June 2024`**: We have released the [Jittor](https://cg.cs.tsinghua.edu.cn/jittor/) version of [Anti-UAV](https://github.com/ZhaoJ9014/Anti-UAV/tree/master/anti_uav_jittor) for domestic hardware support and inference speed up. :blush:

:white_check_mark: **`04 July 2024`**: We have released the [Jittor](https://cg.cs.tsinghua.edu.cn/jittor/) version of [Anti-UAV410](https://github.com/ZhaoJ9014/Anti-UAV/tree/master/anti_uav_jittor) for domestic hardware support and inference speed up. :blush:

****
## Originality
To our best knowledge, we are the first to propose a new Anti-UAV task, corresponding datasets, evaluation metrics and baseline methods, with both PyTorch and Jittor support.

****
## Contents
* [Task Definition](#Task-Definition)
* [Motivation](#Motivation)
* [Environment Setup](#Environment-Setup)
* [Data Preparation](#Data-Preparation)
* [Evaluation Metrics](#Evaluation-Metrics)
* [Training and Inference](#Training-and-Inference)
* [Notebook](#Notebooks)
* [Model Zoo](#Model-Zoo)
* [FAQs](#FAQs)
* [Achievement](#Achievement)
* [Citation](#Citation)


****
## Task Definition
Anti-UAV refers to discovering, detecting, recognizing, and tracking Unmanned Aerial Vehicle (UAV) targets in the wild and simultaneously estimate the tracking states of the targets given RGB and/or Thermal Infrared (IR) videos. When the target disappears, an invisible mark of the target needs to be given. A lot of higher-level applications can be founded upon anti-UAV, such as security of important area, defense against UAV attack, and automated continuous protection from potential threat caused by UAV instrusion.

****


## Motivation
- The Anti-UAV project of Institute of North Electronic Equipment, Beijing, China is proposed to push the frontiers of discovery, detection and tracking of UAVs in the wild.


- Recently, UAV is growing rapidly in a wide range of consumer communications and networks with their autonomy, flexibility, and a broad range of application domains. UAV applications offer possible civil and public domain applications in which single or multiple UAVs may be used. At the same time, we also need to be aware of the potential threat to airspace safty caused by UAV intrusion. Earlier this year, multiple instances of drone sightings halted air traffic at airports, leading to significant economic losses for airlines.


- Currently, in computer vision community, there is no high-quality benchmark for anti-UAV captured in real-world dynamic scenarios. To mitigate this gap, this project presents a new dataset, evaluation metric, and baseline method for the area of discovering, detecting, recognizing, and tracking UAVs. The dataset consists of high quality, Full HD video sequences (both RGB and IR), spanning multiple occurrences of multi-scale UAVs, densely annotated with bounding boxes, attributes, and flags indicating whether the target exists or not in each frame.

****
## Environment Setup
Refer to: [3rd_Anti-UAV_CVPR2023](https://modelscope.cn/models/iic/3rd_Anti-UAV_CVPR23/summary).

Note:

* We recommend that you use `python == 3.8`, or conflicts may occur;
* For those using NVIDIA RTX 30 & 40 series GPUs, the cuda version 11.8 has been tested successfully;
* When you run the command `pip install -r requirements/cv.txt`, there might be some problems, like unable to find bmt_clipit, clip, panopticapi, videofeatures_clipit. And these errors can be ignored;
* You need to install jibjpeg4py library and `jittor == 1.3.8.5`.

****
## Data Preparation

Currently, we offer three public datasets for the ANTI-UAV task.

<img src="https://github.com/ZhaoJ9014/Anti-UAV/blob/master/Fig/1.gif" width="1000px"/>
<div align="center">
    <img src="https://github.com/ZhaoJ9014/Anti-UAV/blob/master/Fig/2.gif" width="400"/><img src="https://github.com/ZhaoJ9014/Anti-UAV/blob/master/Fig/3.gif" width="400"/>
</div>
<div align="center">
  <img src="https://github.com/ZhaoJ9014/Anti-UAV/blob/master/Fig/1.png" width="800px"/>
</div>




- Folder Tree Diagram
<div align="left">
  <img src="https://github.com/ZhaoJ9014/Anti-UAV/blob/master/Fig/folder-tree.png" width="350px"/>
</div>


- Scenario Variations: Compared to the previous challenge, we further enlarge the dataset this year by adding more challenging video sequences with dynamic backgrounds, complex movements, and tiny-scale targets, such that the resulting new dataset covers a greater variety of scenarios with multi-scale UAVs. Examples are shown as follows.
<img src="https://github.com/ZhaoJ9014/Anti-UAV/blob/master/Fig/example.gif" width="1000px"/>

- Download:

| Dataset     | Google Drive | Baidu Drive |
|-------------|--------------|-------------|
| Anti-UAV300 | [link](https://drive.google.com/file/d/1NPYaop35ocVTYWHOYQQHn8YHsM9jmLGr/view)| [Password:sagx](https://pan.baidu.com/s/1dJR0VKyLyiXBNB_qfa2ZrA)   |
| Anti-UAV410 | N/A          | [Password:wfds](https://pan.baidu.com/s/1PbINXhxc-722NWoO8P2AdQ)   |
| Anti-UAV600 | [modelscope](https://modelscope.cn/datasets/ly261666/3rd_Anti-UAV/files)         | N/A         |

* Please note that 410 and 600 versions only contain IR videos while 300 version contains both RGB videos and IR videos. In this release, the model is capable of dealing with both RGB data and IR data, so we recommend that you use the 300 version dataset.

- Please refer to our [Anti-UAV v1 paper](https://ieeexplore.ieee.org/document/9615243) and [Anti-UAV v3 paper](https://arxiv.org/pdf/2306.15767.pdf) for more details ([WeChat News](https://zhaoj9014.github.io/pub/Anti-UAV.pdf)).

****


## Evaluation Metrics
We define the tracking accuracy as:
<img src="https://github.com/ZhaoJ9014/Anti-UAV/blob/master/Fig/3.png" width="1000px"/>
For frame t, IoU_t is Intersection over Union (IoU) between the predicted tracking box and its corresponding ground-truth box, p_t is the predicted visibility flag, it equals 1 when the predicted box is empty and 0 otherwise. The v_t is the ground-truth visibility flag of the target, the indicator function δ(v_t>0) equals 1 when v_t > 0 and 0 otherwise. The accuracy is averaged over all frames in a sequence, T indicates total frames and T^* denotes the number of frames corresponding to the presence of the target in the ground-truth.

****
## Training and Inference
- Training

Currently, we have some problems with training in [Jittor](https://cg.cs.tsinghua.edu.cn/jittor/), but you can still try to run the commands as follows:
```bash
cd {PROJECT_ROOT}/anti_uav_jittor
python ltr/run_training.py modal modal
```

Or you can train the model with [PyTorch](https://pytorch.org).

Also, if you have any suggestions on how to do it, feel free to open an issue!

- Inference

In the root path of the project, run the command `python pysot_toolkit/test.py`
***
## Notebooks
We provide a demo notebook in [`anti_uav_jittor/demo.ipynb`](https://github.com/ZhaoJ9014/Anti-UAV/blob/master/anti_uav_jittor/demo.ipynb) that can help developers better understand the workflow of this demo.

****
## Model Zoo

:monkey:

Keep updating...


****
## FAQs

We will keep updating this section, so feel free to open an issue.
<details>
<summary>1. Q: When I run the training command, there is an error indicating that setting an array element with a sequence, the requested array has an inhomogeneous shape.</summary>
A: In AntiFusion.py, set `visible_data = np.array, dtype = object`.
</details>




****
## Achievement

****
- CVPR 2020 Anti-UAV Workshop & Challenge
<img src="https://github.com/ZhaoJ9014/Anti-UAV/blob/master/Fig/4.png" width="1000px"/>
We have organized the CVPR 2020 Anti-UAV Workshop & Challenge, which is collaborated by INEE, CASIA, TJU, XJTU, Pensees, Xpeng Motors, USTC, NUS, and Baidu.


* [Result Submission & Leaderboard](https://anti-uav.github.io/submission/).  


* [WeChat News1](https://mp.weixin.qq.com/s/SRCf_5L_mzPvV2M9kipUig).


* [WeChat News2](https://blog.csdn.net/m0_46422918/article/details/104551706).


* [WeChat News3](https://mp.weixin.qq.com/s/DwJ8Y4ZIGhgJdowUkl5MiQ).

****

- ICCV 2021 Anti-UAV Workshop & Challenge
<img src="https://github.com/ZhaoJ9014/Anti-UAV/blob/master/Fig/ICCV21.png" width="1000px"/>
We have organized the ICCV 2021 Anti-UAV Workshop & Challenge, which is collaborated by BIT, BUPT, HIT, BJTU, Qihoo 360, OPPO, CAS, and Baidu.


* [Result Submission & Leaderboard](https://anti-uav.github.io/submission/).  


* [WeChat News1](https://mp.weixin.qq.com/s/SRCf_5L_mzPvV2M9kipUig).


* [WeChat News2](https://mp.weixin.qq.com/s/_TQjWGlv5w8EhPgYEZRnnQ).


* [WeChat News3](https://mp.weixin.qq.com/s/ySBtc9pDJBtOcdo12qMXOw).


* [WeChat News4](https://mp.weixin.qq.com/s/3eA8isQ6axgta7bmPgAxZg).


* [WeChat News5](https://mp.weixin.qq.com/s/mQnkEqUvg996oSIfvuQXIA).


* [Qihoo 360 Summary News](https://www.geekpark.net/news/282330).


* [CJIG Summary News](https://mp.weixin.qq.com/s/ZGGjUzn3TDeAV_JeI2j5ZQ).


* [BSIG Summary News](https://mp.weixin.qq.com/s/Z5Qk4QxxRMqPWcbXupXiRQ).

****
- CVPR 2023 Anti-UAV Workshop & Challenge
<img src="https://github.com/ZhaoJ9014/Anti-UAV/blob/master/Fig/CVPR23.png" width="1000px"/>
We have organized the CVPR 2023 Anti-UAV Workshop & Challenge, which is collaborated by BIT, BUPT, HIT, BJTU, Qihoo 360, OPPO, CAS, and Baidu.


* [Result Submission & Leaderboard](https://anti-uav.github.io/submission/).  


* [WeChat News1](https://mp.weixin.qq.com/s/BuK9Lba4taFgEprlbzhAnA).


* [WeChat News2](https://mp.weixin.qq.com/s/YILT9CWKXg5dVuj-ZqVJJg).


* [ModelScope News](https://community.modelscope.cn/63e0d79d406cc115977189e5.html).


* [CSDN News](https://blog.csdn.net/sunbaigui/article/details/128900807).

****
### Citation
- Please consult and consider citing the following papers:


      @article{zhang2023review,
      title={Review and Analysis of RGBT Single Object Tracking Methods: A Fusion Perspective},
      author={Zhang, ZhiHao and Wang, Jun and Zang, Zhuli and Jin, Lei and Li, Shengjie and Wu, Hao and Zhao, Jian and Bo, Zhang},
      journal={T-OMM},
      year={2023}
      }


      @article{huang2023anti,
      title={Anti-UAV410: A Thermal Infrared Benchmark and Customized Scheme for Tracking Drones in the Wild},
      author={Huang, Bo and Li, Jianan and Chen, Junjie and Wang, Gang and Zhao, Jian and Xu, Tingfa},
      journal={T-PAMI},
      year={2023}
      }


      @inproceedings{zhang2023modality,
      title={Modality Meets Long-Term Tracker: A Siamese Dual Fusion Framework for Tracking UAV},
      author={Zhang, Zhihao and Jin, Lei and Li, Shengjie and Xia, JianQiang and Wang, Jun and Li, Zun and Zhu, Zheng and Yang, Wenhan and Zhang, PengFei and Zhao, Jian and others},
      booktitle={ICIP 2023},
      year={2023}
      }


      @article{jiang2021anti,
      title={Anti-UAV: a large-scale benchmark for vision-based UAV tracking},
      author={Jiang, Nan and Wang, Kuiran and Peng, Xiaoke and Yu, Xuehui and Wang, Qiang and Xing, Junliang and Li, Guorong and Ye, Qixiang and Jiao,           Jianbin and Han, Zhenjun and others},
      journal={T-MM},
      year={2021}
      }
      
      
      @article{zhao20212nd,
      title={The 2nd anti-UAV workshop \& challenge: methods and results},
      author={Zhao, Jian and Wang, Gang and Li, Jianan and Jin, Lei and Fan, Nana and Wang, Min and Wang, Xiaojuan and Yong, Ting and Deng, Yafeng and           Guo, Yandong and others},
      journal={arXiv preprint arXiv:2108.09909},
      year={2021}
      }


      @article{zhu2023evidential,
      title={Evidential Detection and Tracking Collaboration: New Problem, Benchmark and Algorithm for Robust Anti-UAV System},
      author={Zhu, Xue-Feng and Xu, Tianyang and Zhao, Jian and Liu, Jia-Wei and Wang, Kai and Wang, Gang and Li, Jianan and Zhang, Zhihao and Wang, Qiang and Jin, Lei and     others},
      journal={arXiv preprint arXiv:2306.15767},
      year={2023}
      }


      @article{zhao20233rd,
      title={The 3rd Anti-UAV Workshop \& Challenge: Methods and Results},
      author={Zhao, Jian and Li, Jianan and Jin, Lei and Chu, Jiaming and Zhang, Zhihao and Wang, Jun and Xia, Jiangqiang and Wang, Kai and Liu, Yang and Gulshad, Sadaf and others},
      journal={arXiv preprint arXiv:2305.07290},
      year={2023}
      }


      @article{zhao20233rd,
      title={The 3rd Anti-UAV Workshop \& Challenge: Methods and Results},
      author={Zhao, Jian and Li, Jianan and Jin, Lei and Chu, Jiaming and Zhang, Zhihao and Wang, Jun and Xia, Jiangqiang and Wang, Kai and Liu, Yang and Gulshad, Sadaf and others},
      journal={arXiv preprint arXiv:2305.07290},
      year={2023}
      }
