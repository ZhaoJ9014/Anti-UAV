from got10k_toolkit.toolkit.experiments import ExperimentGOT10k

#Specify the path
dataset_root= '' #Absolute path of the datasets

#Evaluation
experiment = ExperimentGOT10k(
    root_dir=dataset_root,  # GOT-10k's root directory
    subset='test',  # 'train' | 'val' | 'test'
    result_dir='results',  # where to store tracking results
    report_dir='reports'  # where to store evaluation reports
)
experiment.report(['transt'])