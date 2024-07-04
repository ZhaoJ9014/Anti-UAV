from __future__ import absolute_import

import unittest
import os

from trackers.SiamFC.siamfc import TrackerSiamFC
from experiments import ExperimentAntiUAV410


"""
Experiments Setup
"""

dataset_path='path/to/Anti-UAV410'

# test or val
subset='test'

net_path = './Trackers/SiamFC/model.pth'
tracker = TrackerSiamFC(net_path=net_path)

# run experiment
experiment = ExperimentAntiUAV410(root_dir=dataset_path, subset=subset)

experiment.run(tracker, visualize=True)
# report performance
experiment.report([tracker.name])