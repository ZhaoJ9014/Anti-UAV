# Anti-UAV410 Benchmark

Anti-UAV410: A Thermal Infrared Benchmark and Customized Scheme for Tracking Drones in the Wild

This toolkit is used to evaluate trackers on generalized infrared UAV tracking benchmark called Anti-UAV410. The benchmark comprises a total of 410 videos with over 438K manually annotated bounding boxes.

# News

* The **SiamDT Tracker** has been released! Please refer to path [trackers/SiamDT/](trackers/SiamDT/README.md).

* The **Matlab version of the AntiUAV410 benchmark** has been released! Please refer to path [toolkit_matlab/](toolkit_matlab/).

* The **Python version of the AntiUAV410 benchmark** has been released!

## Preparing the dataset
Download the Anti-UAV410 dataset ([Google drive](https://drive.google.com/file/d/1zsdazmKS3mHaEZWS2BnqbYHPEcIaH5WR/view?usp=sharing) and [Baidu disk](https://pan.baidu.com/s/1R-L9gKIRowMgjjt52n48-g?pwd=a410) Access code: a410) to your disk, the organized directory should look like:

    ```
    --AntiUAV410/
    	|--test
    	|--train
    	|--val
    ```

note that the annotations for each video attribute are under the annos/test/att or annos/train/att or annos/val/att paths.

## Installation and testing
**Step 1.** Create a conda environment and activate it.

```shell
conda create -n AntiUAV410 python=3.9.12
conda activate AntiUAV410
```

**Step 2.** Install the requirements.
```shell
pip install opencv-python, matplotlib, wget, shapely

pip install torch===1.9.1 -f https://download.pytorch.org/whl/torch_stable.html
pip install torchvision===0.10.1 -f https://download.pytorch.org/whl/torch_stable.html
```
Other versions of python, cuda and torch are also compatible.

**Step 3.** Testing the default SiamFC tracker.

Change the dataset_path in the Demo_for_tracking.py file to the path where the dataset is located.

Run
```shell
python Demo_for_tracking.py
```
The tracking results will be saved at project_dir/results/AntiUAV410/test/SiamFC.

**Step 4.** Install the SiamDT tracker.

Please refer to the [installation tutorials](trackers/SiamDT/README.md).

**Step 5.** Downloading the tracking results compared in the paper.

Download the tracking results ([Google drive](https://drive.google.com/file/d/1AlLpoMorj-7bKA1zqo1DkuEZ9h0jQs_-/view?usp=sharing) and [Baidu disk](https://pan.baidu.com/s/12NRarQvIiyZKbXRu5fPGpw?pwd=a410) Access code: a410) to your project directory, the organized directory should look like:

    ```
    --project_dir/tracking_results/
    	|--Defaults
    	|--Trained_with_antiuav410
    ```

The files inside the Defaults directory are the results of the trackers that are not trained with Anti-UAV410 dataset, while The files inside the Trained_with_antiuav410 directory are the results of the trackers that are re-trained with Anti-UAV410 dataset.

**Step 6.** Visual comparison.

Change the dataset path and select the trackers that need to be compared visually.

Run
```shell
python Demo_for_visual_comparison.py
```

The comparison figures will be saved at project_dir/figures/.
<!---
![contents](./figures/02_6319_1500-2999.jpg)
-->
<img src="figures/02_6319_1500-2999.jpg" width="30%"><img src="figures/3700000000002_144152_1.jpg" width="30%"><img src="figures/3700000000002_152538_1.jpg" width="30%">


``not exist'' in the figure means that the target is occluded or out of view.

**Step 7.** Evaluating the trackers.

Change the dataset path and edit project_dir/utils/trackers.py to select the trackers to be evaluated.

Run
```shell
python Evaluation_for_ALL.py
```

The evaluation plots will be saved at project_dir/reports/AntiUAV410/.

Over 50 trackers are involved, they are:
### Default trackers

<img src="figures/success_plots_Default.png" width="90%">

* **MixFormerV2-B.** Cui, Yutao, et al. "Mixformerv2: Efficient fully transformer tracking." NIPS, 2023. [[Github]](https://github.com/MCG-NJU/MixFormerV2)
* **ROMTrack.** Cai, Yidong, et al. "Robust object modeling for visual tracking." ICCV, 2023. [[Github]](https://github.com/dawnyc/ROMTrack)
* **GRM.** Gao, Shenyuan, et al. "Generalized relation modeling for transformer tracking." CVPR, 2023. [[Github]](https://github.com/Little-Podi/GRM)
* **DropTrack.**  Wu, Qiangqiang, et al. "Dropmae: Masked autoencoders with spatial-attention dropout for tracking tasks." CVPR, 2023. [[Github]](https://github.com/jimmy-dq/DropMAE)
* **ARTrack.** Wei, Xing, et al. "Autoregressive visual tracking." CVPR, 2023. [[Github]](https://github.com/MIV-XJTU/ARTrack)
* **SeqTrack-B256.** Chen, Xin, et al. "Seqtrack: Sequence to sequence learning for visual object tracking." CVPR, 2023. [[Github]](https://github.com/microsoft/VideoX)
* **SeqTrack-B384.** Chen, Xin, et al. "Seqtrack: Sequence to sequence learning for visual object tracking." CVPR, 2023. [[Github]](https://github.com/microsoft/VideoX)
* **JointNLT.** Zhou, Li, et al. "Joint visual grounding and tracking with natural language specification." CVPR, 2023. [[Github]](https://github.com/lizhou-cs/JointNLT)
* **SwinTrack-Tiny.** Lin, Liting, et al. "Swintrack: A simple and strong baseline for transformer tracking." NIPS, 2022. [[Github]](https://github.com/LitingLin/SwinTrack)
* **SwinTrack-Base.** Lin, Liting, et al. "Swintrack: A simple and strong baseline for transformer tracking." NIPS, 2022. [[Github]](https://github.com/LitingLin/SwinTrack)
* **ToMP50.** Mayer, Christoph, et al. "Transforming model prediction for tracking." CVPR, 2022. [[Github]](https://github.com/visionml/pytracking)
* **ToMP101.** Mayer, Christoph, et al. "Transforming model prediction for tracking." CVPR, 2022. [[Github]](https://github.com/visionml/pytracking)
* **TCTrack.** Cao, Ziang, et al. "Tctrack: Temporal contexts for aerial tracking." CVPR, 2022. [[Github]](https://github.com/vision4robotics/TCTrack)
* **SLT-TransT.** Kim, Minji, et al. "Towards sequence-level training for visual tracking." ECCV, 2022. [[Github]](https://github.com/byminji/SLTtrack)
* **OSTrack-256.** Ye, Botao, et al. "Joint feature learning and relation modeling for tracking: A one-stream framework." ECCV, 2022. [[Github]](https://github.com/botaoye/OSTrack)
* **OSTrack-384.** Ye, Botao, et al. "Joint feature learning and relation modeling for tracking: A one-stream framework." ECCV, 2022. [[Github]](https://github.com/botaoye/OSTrack)
* **AiATrack.** Gao, Shenyuan, et al. "Aiatrack: Attention in attention for transformer visual tracking." ECCV, 2022. [[Github]](https://github.com/Little-Podi/AiATrack)
* **Unicorn-Tiny.** Yan, Bin, et al. "Towards grand unification of object tracking." ECCV, 2022. [[Github]](https://github.com/MasterBin-IIAU/Unicorn)
* **Unicorn-Large.** Yan, Bin, et al. "Towards grand unification of object tracking." ECCV, 2022. [[Github]](https://github.com/MasterBin-IIAU/Unicorn)
* **RTS.** Paul, Matthieu, et al. "Robust visual tracking by segmentation." ECCV, 2022. [[Github]](https://github.com/visionml/pytracking)
* **KeepTrack.** Mayer, Christoph, et al. "Learning target candidate association to keep track of what not to track." ICCV, 2021. [[Github]](https://github.com/visionml/pytracking)
* **Stark-ST50.** Yan, Bin, et al. "Learning spatio-temporal transformer for visual tracking." ICCV, 2021. [[Github]](https://github.com/researchmm/Stark)
* **Stark-ST101.** Yan, Bin, et al. "Learning spatio-temporal transformer for visual tracking." ICCV, 2021. [[Github]](https://github.com/researchmm/Stark)
* **HiFT.** Cao, Ziang, et al. "Hift: Hierarchical feature transformer for aerial tracking." ICCV, 2021. [[Github]](https://github.com/vision4robotics/HiFT)
* **STMTrack.** Fu, Zhihong, et al. "Stmtrack: Template-free visual tracking with space-time memory networks." CVPR, 2021. [[Github]](https://github.com/fzh0917/STMTrack)
* **TrDiMP.** Wang, Ning, et al. "Transformer meets tracker: Exploiting temporal context for robust visual tracking." CVPR, 2021. [[Github]](https://github.com/594422814/TransformerTrack)
* **TransT.** Chen, Xin, et al. "Transformer tracking." CVPR, 2021. [[Github]](https://github.com/chenxin-dlut/TransT)
* **ROAM.** Yang, Tianyu, et al. "ROAM: Recurrently optimizing tracking model." CVPR, 2020. [[Github]](https://github.com/tyyyang/ROAM)
* **Siam R-CNN.** Voigtlaender, Paul, et al. "Siam r-cnn: Visual tracking by re-detection." CVPR, 2020. [[Github]](https://www.vision.rwth-aachen.de/page/siamrcnn)
* **SiamBAN.** Chen, Zedu, et al. "Siamese box adaptive network for visual tracking." CVPR, 2020. [[Github]](https://github.com/hqucv/siamban)
* **SiamCAR.** Guo, Dongyan, et al. "SiamCAR: Siamese fully convolutional classification and regression for visual tracking." CVPR, 2020. [[Github]](https://github.com/ohhhyeahhh/SiamCAR)
* **GlobalTrack.** Huang, Lianghua, et al. "Globaltrack: A simple and strong baseline for long-term tracking." AAAI, 2020. [[Github]](https://github.com/huanglianghua/GlobalTrack)
* **KYS.** Bhat, Goutam, et al. "Know your surroundings: Exploiting scene information for object tracking." ECCV, 2020. [[Github]](https://github.com/visionml/pytracking)
* **Super DiMP.** -- -- --. [[Github]](https://github.com/visionml/pytracking)
* **PrDiMP50.** Danelljan, Martin, et al. "Probabilistic regression for visual tracking." CVPR, 2020. [[Github]](https://github.com/visionml/pytracking)
* **SiamFC++.** Xu, Yinda, et al. "Siamfc++: Towards robust and accurate visual tracking with target estimation guidelines." AAAI, 2020. [[Github]](https://github.com/megvii-research/video_analyst)

* **and so on.**

### Re-trained trackers

<img src="figures/success_plots_Re-trained.png" width="90%">

* **MixFormerV2-B.** Cui, Yutao, et al. "Mixformerv2: Efficient fully transformer tracking." NIPS, 2023. [[Github]](https://github.com/MCG-NJU/MixFormerV2)
* **DropTrack.**  Wu, Qiangqiang, et al. "Dropmae: Masked autoencoders with spatial-attention dropout for tracking tasks." CVPR, 2023. [[Github]](https://github.com/jimmy-dq/DropMAE)
* **SwinTrack-Tiny.** Lin, Liting, et al. "Swintrack: A simple and strong baseline for transformer tracking." NIPS, 2022. [[Github]](https://github.com/LitingLin/SwinTrack)
* **SwinTrack-Base.** Lin, Liting, et al. "Swintrack: A simple and strong baseline for transformer tracking." NIPS, 2022. [[Github]](https://github.com/LitingLin/SwinTrack)
* **ToMP50.** Mayer, Christoph, et al. "Transforming model prediction for tracking." CVPR, 2022. [[Github]](https://github.com/visionml/pytracking)
* **ToMP101.** Mayer, Christoph, et al. "Transforming model prediction for tracking." CVPR, 2022. [[Github]](https://github.com/visionml/pytracking)
* **TCTrack.** Cao, Ziang, et al. "Tctrack: Temporal contexts for aerial tracking." CVPR, 2022. [[Github]](https://github.com/vision4robotics/TCTrack)
* **AiATrack.** Gao, Shenyuan, et al. "Aiatrack: Attention in attention for transformer visual tracking." ECCV, 2022. [[Github]](https://github.com/Little-Podi/AiATrack)
* **KeepTrack.** Mayer, Christoph, et al. "Learning target candidate association to keep track of what not to track." ICCV, 2021. [[Github]](https://github.com/visionml/pytracking)
* **Stark-ST101.** Yan, Bin, et al. "Learning spatio-temporal transformer for visual tracking." ICCV, 2021. [[Github]](https://github.com/researchmm/Stark)
* **Siam R-CNN.** Voigtlaender, Paul, et al. "Siam r-cnn: Visual tracking by re-detection." CVPR, 2020. [[Github]](https://www.vision.rwth-aachen.de/page/siamrcnn)
* **SiamBAN.** Chen, Zedu, et al. "Siamese box adaptive network for visual tracking." CVPR, 2020. [[Github]](https://github.com/hqucv/siamban)
* **GlobalTrack.** Huang, Lianghua, et al. "Globaltrack: A simple and strong baseline for long-term tracking." AAAI, 2020. [[Github]](https://github.com/huanglianghua/GlobalTrack)
* **KYS.** Bhat, Goutam, et al. "Know your surroundings: Exploiting scene information for object tracking." ECCV, 2020. [[Github]](https://github.com/visionml/pytracking)
* **Super DiMP.** -- -- --. [[Github]](https://github.com/visionml/pytracking)
* **PrDiMP50.** Danelljan, Martin, et al. "Probabilistic regression for visual tracking." CVPR, 2020. [[Github]](https://github.com/visionml/pytracking)


### Analysis of attributes
The annotations for each video attribute are under the annos/test/att or annos/train/att or annos/val/att paths, and the attributes in order are ['Thermal Crossover', 'Out-of-View', 'Scale Variation', 'Fast Motion', 'Occlusion', 'Dynamic Background Clutter', 'Tiny Size', 'Small Size', 'Medium Size', 'Normal Size']. The attributes for each frame are labelled in IR_label.json, but it is not complete.

<img src="figures/att/success_plots_of_TC.png" width="30%"><img src="figures/att/success_plots_of_OV.png" width="30%"><img src="figures/att/success_plots_of_SV.png" width="30%">
<img src="figures/att/success_plots_of_FM.png" width="30%"><img src="figures/att/success_plots_of_OC.png" width="30%"><img src="figures/att/success_plots_of_DBC.png" width="30%">
<img src="figures/att/success_plots_of_TS.png" width="30%"><img src="figures/att/success_plots_of_SS.png" width="30%"><img src="figures/att/success_plots_of_MS.png" width="30%">
<img src="figures/att/success_plots_of_NS.png" width="30%">


## Citation

If you find this project useful in your research, please consider cite:

```latex
@article{huang2023anti,
  title={Anti-UAV410: A Thermal Infrared Benchmark and Customized Scheme for Tracking Drones in the Wild},
  author={Huang, Bo and Li, Jianan and Chen, Junjie and Wang, Gang and Zhao, Jian and Xu, Tingfa},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  year={2023},
  publisher={IEEE}
}
```
