# Anti-Unmanned Aerial Vehicle (UAV)
* This work was done during Jian Zhao served as an assistant professor at Institute of North Electronic Equipment, Beijing, China.

|Author|Jian Zhao|
|:---:|:---:|
|Homepage|https://zhaoj9014.github.io|

****
## License

The project of anti-UAV is released under the MIT License.

****


## Originality
- To our best knowledge, we are the first to propose a new anti-UAV task, corresponding datasets, evaluation metrics and baseline methods.

****


## Task Definition
- Anti-UAV refers to discovering, detecting, recognizing, and tracking Unmanned Aerial Vehicle (UAV) targets in the wild and simultaneously estimate the tracking states of the targets given RGB and/or Thermal Infrared (IR) videos. When the target disappears, an invisible mark of the target needs to be given. A lot of higher-level applications can be founded upon anti-UAV, such as security of important area, defense against UAV attack, and automated continuous protection from potential threat caused by UAV instrusion.

****


## Motivation
- The anti-UAV project of Institute of North Electronic Equipment, Beijing, China is proposed to push the frontiers of discovering, detection and tracking of UAVs in the wild.


- Recently, UAV is growing rapidly in a wide range of consumer communications and networks with their autonomy, flexibility, and a broad range of application domains. UAV applications offer possible civil and public domain applications in which single or multiple UAVs may be used. At the same time, we also need to be aware of the potential threat to airspace safty caused by UAV intrusion. Earlier this year, multiple instances of drone sightings halted air traffic at airports, leading to significant economic losses for airlines.


- Currently, in computer vision community, there is no high-quality benchmark for anti-UAV captured in real-world dynamic scenarios. To mitigate this gap, this project presents a new dataset, evaluation metric, and baseline method for the area of discovering, detecting, recognizing, and tracking UAVs. The dataset consists of high quality, Full HD video sequences (both RGB and IR), spanning multiple occurrences of multi-scale UAVs, densely annotated with bounding boxes, attributes, and flags indicating whether the target exists or not in each frame.

****


## Anti-UAV Dataset
<img src="https://github.com/ZhaoJ9014/Anti-UAV/blob/master/Fig/1.gif" width="1000px"/>
<div align="center">
    <img src="https://github.com/ZhaoJ9014/Anti-UAV/blob/master/Fig/2.gif" width="400"/><img src="https://github.com/ZhaoJ9014/Anti-UAV/blob/master/Fig/3.gif" width="400"/>
</div>
<div align="center">
  <img src="https://github.com/ZhaoJ9014/Anti-UAV/blob/master/Fig/1.png" width="800px"/>
</div>


- There are three subsets in the dataset, i.e., the train subset, the test subset for track 1 and the test subset for track 2. The train subset consists of 200 thermal infrared video sequences and publishes detailed annotation files (whether the target exists, target location information and various challenges). The subset for track 1 also contains 200 video sequences, only providing the position information of target in the first frame; The subset for track 2 contains 200 video sequences. This track does not provide any labeled information. It requires participants to obtain the flag of existence and corresponding target location information of the target through detection and tracking. Above three subsets do not have any overlap between each other. We propose participants could train a suitable detector or tracker model depending on multiple categories of label information in train subset.


- Folder Tree Diagram
<div align="left">
  <img src="https://github.com/ZhaoJ9014/Anti-UAV/blob/master/Fig/folder-tree.png" width="350px"/>
</div>


- Scenario Variations: Compared to the previous challenge, we further enlarge the dataset this year by adding more challenging video sequences with dynamic backgrounds, complex movements, and tiny-scale targets, such that the resulting new dataset covers a greater variety of scenarios with multi-scale UAVs. Examples are shown as follows.
<img src="https://github.com/ZhaoJ9014/Anti-UAV/blob/master/Fig/example.gif" width="1000px"/>

- Download: The anti-UAV dataset is available at [google drive](https://drive.google.com/open?id=1GICr5e9CZN0tcFM_VXhyogzxWD3LMvAw) (v1) and [baidu drive](https://pan.baidu.com/s/1dJR0VKyLyiXBNB_qfa2ZrA) (password: sagx) (v1) / [baidu drive](https://pan.baidu.com/s/1PbINXhxc-722NWoO8P2AdQ) (password: wfds) (v2).

- Please refer to our [Anti-UAV paper](https://arxiv.org/pdf/2101.08466.pdf) for more details ([WeChat News](https://zhaoj9014.github.io/pub/Anti-UAV.pdf)).

****


## Evaluation Metrics
- We define the tracking accuracy as:
<img src="https://github.com/ZhaoJ9014/Anti-UAV/blob/master/Fig/3.png" width="1000px"/>
For frame t, IoU_t is Intersection over Union (IoU) between the predicted tracking box and its corresponding ground-truth box, p_t is the predicted visibility flag, it equals 1 when the predicted box is empty and 0 otherwise. The v_t is the ground-truth visibility flag of the target, the indicator function δ(v_t>0) equals 1 when v_t > 0 and 0 otherwise. The accuracy is averaged over all frames in a sequence, T indicates total frames and T^* denotes the number of frames corresponding to the presence of the target in the ground-truth.


## CVPR 2020 Anti-UAV Workshop & Challenge
<img src="https://github.com/ZhaoJ9014/Anti-UAV/blob/master/Fig/4.png" width="1000px"/>
- We have organized the CVPR 2020 Anti-UAV Workshop & Challenge, which is collaborated by INEE, CASIA, TJU, XJTU, Pensees, Xpeng Motors, USTC, NUS, and Baidu.


- [Result Submission & Leaderboard](https://anti-uav.github.io/submission/).  


- [WeChat News1](https://mp.weixin.qq.com/s/SRCf_5L_mzPvV2M9kipUig).


- [WeChat News2](https://blog.csdn.net/m0_46422918/article/details/104551706).


- [WeChat News3](https://mp.weixin.qq.com/s/DwJ8Y4ZIGhgJdowUkl5MiQ).

****


## ICCV 2021 Anti-UAV Workshop & Challenge
<img src="https://github.com/ZhaoJ9014/Anti-UAV/blob/master/Fig/ICCV21.png" width="1000px"/>
- We have organized the ICCV 2021 Anti-UAV Workshop & Challenge, which is collaborated by BIT, BUPT, HIT, BJTU, Qihoo 360, OPPO, CAS, and Baidu.


- [Result Submission & Leaderboard](https://anti-uav.github.io/submission/).  


- [WeChat News1](https://mp.weixin.qq.com/s/SRCf_5L_mzPvV2M9kipUig).


- [WeChat News2](https://mp.weixin.qq.com/s/_TQjWGlv5w8EhPgYEZRnnQ).


- [WeChat News3](https://mp.weixin.qq.com/s/ySBtc9pDJBtOcdo12qMXOw).


- [WeChat News4](https://mp.weixin.qq.com/s/3eA8isQ6axgta7bmPgAxZg).


- [WeChat News5](https://mp.weixin.qq.com/s/mQnkEqUvg996oSIfvuQXIA).


- [Qihoo 360 Summary News](https://www.geekpark.net/news/282330).


- [CJIG Summary News](https://mp.weixin.qq.com/s/ZGGjUzn3TDeAV_JeI2j5ZQ).


- [BSIG Summary News](https://mp.weixin.qq.com/s/Z5Qk4QxxRMqPWcbXupXiRQ).

****


## CVPR 20213 Anti-UAV Workshop & Challenge
<img src="https://github.com/ZhaoJ9014/Anti-UAV/blob/master/Fig/CVPR23.png" width="1000px"/>
- We will organized the CVPR 2023 Anti-UAV Workshop & Challenge, which is collaborated by BIT, BUPT, HIT, BJTU, Qihoo 360, OPPO, CAS, and Baidu.


- [Result Submission & Leaderboard](https://anti-uav.github.io/submission/).  


- [WeChat News1](https://mp.weixin.qq.com/s/BuK9Lba4taFgEprlbzhAnA).


### Citation
- Please consult and consider citing the following papers:


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
