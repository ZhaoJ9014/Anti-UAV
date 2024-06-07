from got10k_toolkit.toolkit.experiments import ExperimentNfS

dataset_root= '' #Absolute path of the dataset

#Evaluation
experiment = ExperimentNfS(
    root_dir=dataset_root,  # GOT-10k's root directory
    fps=30,
    result_dir='results',  # where to store tracking results
    report_dir='reports'  # where to store evaluation reports
)
experiment.report(['transt'])