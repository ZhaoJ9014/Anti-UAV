from got10k_toolkit.toolkit.experiments import ExperimentTrackingNet

#Specify the path
dataset_root= '' #Absolute path of the datasets

#Evaluation
experiment = ExperimentTrackingNet(
    root_dir=dataset_root,  # TrackingNet's root directory
    subset='test',  # 'train' | 'val' | 'test'
    result_dir='results',  # where to store tracking results
    report_dir='reports'  # where to store evaluation reports
)
experiment.report(['transt'])