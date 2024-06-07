from got10k_toolkit.toolkit.experiments import ExperimentGOT10k

report_files = ['']
tracker_names = ['']

# setup experiment and plot curves
experiment = ExperimentGOT10k('', subset='test')
experiment.plot_curves(report_files, tracker_names)