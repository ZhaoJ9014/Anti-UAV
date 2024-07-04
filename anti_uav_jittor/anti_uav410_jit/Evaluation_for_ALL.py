from experiments import ExperimentAntiUAV410

from utils.trackers import Default_Trackers as Trackers
# from utils.trackers import Trained_Trackers as Trackers

evaluation_metrics=['State accuracy', 'Success plots', 'Precision plots']

dataset_path='path/to/Anti-UAV410'

# test or val
subset='test'

# Setting experimental parameters
experiment = ExperimentAntiUAV410(root_dir=dataset_path, subset=subset)

experiment.report(Trackers)
