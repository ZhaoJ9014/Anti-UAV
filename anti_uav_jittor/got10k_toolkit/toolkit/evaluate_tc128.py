from got10k_toolkit.toolkit.experiments import ExperimentTColor128

#Specify the path
dataset_root= '' #Absolute path of the datasets

#Evaluation
experiment = ExperimentTColor128(
    root_dir=dataset_root,  # TC128's root directory
    result_dir='results',  # where to store tracking results
    report_dir='reports'  # where to store evaluation reports
)
experiment.report(['transt'])