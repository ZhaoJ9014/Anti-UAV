from got10k_toolkit.toolkit.experiments import ExperimentLaSOT

#Specify the path
dataset_root= '' #Absolute path of the dataset

#Evaluation
experiment = ExperimentLaSOT(
    root_dir=dataset_root,  # LaSOT's root directory
    subset='test',  # 'train' | 'val' | 'test'
    result_dir='results',  # where to store tracking results
    report_dir='reports'  # where to store evaluation reports
)
experiment.report(['transt'])