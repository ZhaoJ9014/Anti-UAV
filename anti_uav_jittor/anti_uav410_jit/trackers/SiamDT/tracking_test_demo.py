import init_paths
import libs.data as data
from trackers import *

# conda activate SiamDT
# python -m visdom.server -port=5123
# python tracking_test_demo.py

if __name__ == '__main__':
    cfg_file = 'configs/siamdt_swin_tiny_sgd.py'
    ckp_file = 'checkpoints/siamdt_swin_tiny_sgd.pth'
    name_suffix = cfg_file[8:-3]
    visualize = False
    # selected_seq='02_6321_0274-2773'
    selected_seq = 'ALL'

    transforms = data.BasicPairTransforms(train=False)
    tracker = SiamDTTracker(
        cfg_file, ckp_file, transforms,
        name_suffix=name_suffix, visualize=visualize)

    evaluators = [
        data.EvaluatorUAVtir(root_dir='/media/data2/TrackingDatasets/Anti-UAV410/Anti-UAV/', subset='test')]

    for e in evaluators:
        e.run(tracker, selected_seq=selected_seq)
