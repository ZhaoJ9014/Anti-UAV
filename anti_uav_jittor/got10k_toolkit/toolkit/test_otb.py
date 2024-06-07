from got10k_toolkit.toolkit.experiments import ExperimentOTB
from got10k_toolkit.toolkit.trackers.identity_tracker import IdentityTracker
from got10k_toolkit.toolkit.trackers.net_wrappers import NetWithBackbone

#Specify the path
net_path = 'D:\study\\track\model\TransT_ep0008_ir.pth.tar' #Absolute path of the model
dataset_root= 'D:\study\\track\dataset\OTB2015\\tb100' #Absolute path of the datasets

#TransT
net = NetWithBackbone(net_path=net_path, use_gpu=True)
tracker = IdentityTracker(name='transt', net=net, window_penalty=0.49, exemplar_size=128, instance_size=256)

#Test
experiment = ExperimentOTB(
    root_dir=dataset_root,  # OTB100's root directory
    result_dir='results',  # where to store tracking results
    report_dir='reports'  # where to store evaluation reports
)
experiment.run(tracker, visualize=False)