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


- Statistics: The anti-UAV dataset consists of 160 high quality, full HD video sequences (both RGB and IR), spanning multiple occurrences of multi-scale UAVs (3 sizes, *i.e.*, large, normal, tiny; 4 models, *i.e.*, DJI-Inspire, DJI-Phantom4, DJI-MarvicAir, DJI-MarvicPRO). The RGB and IR videos are captured by static special ground camera with an automatic rotation platform which can be remotely controlled by PC. All data are densely annotated with bounding boxes, attributes (large, normal, tiny, day, night, cloud, building, false object, speed change, hang, occlusion, scale variation), and flags indicating whether the target exists or not in each frame by professional data annotators. 
<div align="center">
  <img src="https://github.com/ZhaoJ9014/Anti-UAV/blob/master/Fig/2.png" width="500px"/>
</div>


- Download: The anti-UAV dataset is available at [google drive](https://drive.google.com/open?id=1GICr5e9CZN0tcFM_VXhyogzxWD3LMvAw) and [baidu drive](https://pan.baidu.com/s/1dJR0VKyLyiXBNB_qfa2ZrA) (password: sagx).

- Please refer to our [Anti-UAV paper](https://arxiv.org/pdf/2101.08466.pdf) for more details.

****


## Evaluation Metrics
- We define the tracking accuracy as:
<img src="https://github.com/ZhaoJ9014/Anti-UAV/blob/master/Fig/3.png" width="1000px"/>
The IoU_i is Intersection over Union (IoU) between each corresponding ground truth and tracking boxes and the v are the visibility flags of the ground truth (the tracker's predicted p are used to measure the state accuracy). The accuracy is averaged over all frames.


- Note: We provide both RGB and IR videos and their corresponding ground-truths. Challenge participants can only use both IR and RGB videos and their ground-truth location in the first frame. The final evaluation ranks are calculated according to the results on the IR data.


- Test
    - Set up the environment
        ```shell
            conda create -n anti_uav python=3.7
            conda activate anti_uav
            pip install opencv-python torch
        ```
    - Run
        ```shell
            python test.py
        ```
    - You will see the following results
        ```shell
            [001/100]  20190925_131530_1_4    IR Fixed Measure: 0.187
            [002/100]  20190926_183400_1_8    IR Fixed Measure: 0.788
            ...
            [100/100]  20190925_213001_1_2    IR Fixed Measure: 0.028
            [Overall]    IR Mixed Measure: 0.420
        ``` 
        

- Submit Result to codalab
    ```shell
        cd result/SiamFC
        zip -r ../SiamFC_test_dev.zip *.txt
    ``` 
    You can upload the SiamFC_test_dev.zip file to [CodaLab](https://competitions.codalab.org/competitions/23887).

****


## CVPR 2020 Anti-UAV Workshop & Challenge
<img src="https://github.com/ZhaoJ9014/Anti-UAV/blob/master/Fig/4.png" width="1000px"/>
- We will organise the CVPR 2020 Anti-UAV Workshop & Challenge, which is collaborated by INEE, CASIA, TJU, XJTU, Pensees, Xpeng Motors, USTC, NUS, and Baidu.


- [Result Submission & Leaderboard](https://anti-uav.github.io/submission/).  


- [WeChat News1](https://mp.weixin.qq.com/s/SRCf_5L_mzPvV2M9kipUig).


- [WeChat News2](https://blog.csdn.net/m0_46422918/article/details/104551706).


- [WeChat News3](https://mp.weixin.qq.com/s/DwJ8Y4ZIGhgJdowUkl5MiQ).

****


### Citation
- Please consult and consider citing the following papers:


      @article{jiang2021anti,
      title={Anti-UAV: A Large Multi-Modal Benchmark for UAV Tracking},
      author={Jiang, Nan and Wang, Kuiran and Peng, Xiaoke and Yu, Xuehui and Wang, Qiang and Xing, Junliang and Li, Guorong and Guo, Guodong and Zhao, Jian and Han, Zhenjun},
      journal={arXiv preprint arXiv:2101.08466},
      year={2021}
      }
