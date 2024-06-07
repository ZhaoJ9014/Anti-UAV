from got10k_toolkit.toolkit.experiments import ExperimentUAV123

report_files = ['','']
tracker_names = ['','']

# setup experiment and plot curves
experiment = ExperimentUAV123(
    root_dir='',  # UAV123's root directory
    result_dir='results',  # where to store tracking results
    report_dir='reports'  # where to store evaluation reports
)
experiment.plot_curves(tracker_names)