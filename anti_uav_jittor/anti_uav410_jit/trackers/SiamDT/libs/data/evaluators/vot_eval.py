import time
import numpy as np
import os
import glob
import warnings
import json

import libs.ops as ops
from libs.config import registry
from libs.data.datasets import VOT
from .evaluator import Evaluator


__all__ = ['VOT_Eval', 'EvaluatorVOT']


@registry.register_module
class VOT_Eval(Evaluator):
    r"""Evaluation pipeline and evaluation toolkit for VOT-like datasets.

    Notes:
        - The tracking results of three types of experiments ``supervised``
            ``unsupervised`` and ``realtime`` are compatible with the official
            VOT toolkit <https://github.com/votchallenge/vot-toolkit/>`.
        - TODO: The evaluation function for VOT tracking results is still
            under development.
    
    Args:
        dataset (Dataset): A VOT-like dataset.
        experiments (string or list): Specify the type(s) of experiments to run.
            Default is a list [``supervised``, ``unsupervised``, ``realtime``].
        tags (list): list of attribute tags to report.
        result_dir (string, optional): Directory for storing tracking
            results. Default is ``./results``.
        report_dir (string, optional): Directory for storing performance
            evaluation results. Default is ``./reports``.
    """
    def __init__(self, dataset,
                 experiments=['supervised', 'unsupervised', 'realtime'],
                 tags=['camera_motion', 'illum_change', 'occlusion',
                       'size_change', 'motion_change', 'empty'],
                 skip_initialize=5, burnin=10, repetitions=15,
                 sensitive=100, nbins_eao=1500,
                 result_dir='results', report_dir='reports'):
        if isinstance(experiments, str):
            experiments = [experiments]
        assert all([e in ['supervised', 'unsupervised', 'realtime']
                    for e in experiments])
        self.dataset = dataset
        self.experiments = experiments
        self.tags = tags
        self.skip_initialize = skip_initialize
        self.burnin = burnin
        self.repetitions = repetitions
        self.sensitive = sensitive
        self.nbins_eao = nbins_eao
        self.result_dir = os.path.join(result_dir, dataset.name)
        self.report_dir = os.path.join(report_dir, dataset.name)

    def run(self, tracker, visualize=False):
        ops.sys_print('Running tracker %s on %s...' % (
            tracker.name, self.dataset.name))

        # run all specified experiments
        if 'supervised' in self.experiments:
            self.run_supervised(tracker, visualize)
        if 'unsupervised' in self.experiments:
            self.run_unsupervised(tracker, visualize)
        if 'realtime' in self.experiments:
            self.run_realtime(tracker, visualize)

    def run_supervised(self, tracker, visualize=False):
        ops.sys_print('Running supervised experiment...')

        # loop over the complete dataset
        for s, (img_files, target) in enumerate(self.dataset):
            seq_name = self.dataset.seq_names[s]
            ops.sys_print('--Sequence %d/%d: %s' % (
                s + 1, len(self.dataset), seq_name))

            # rectangular bounding boxes
            anno = target['anno']
            anno_rects = anno.copy()
            if anno_rects.shape[1] == 8:
                anno_rects = self.dataset._corner2rect(anno_rects)

            # run multiple repetitions for each sequence
            for r in range(self.repetitions):
                # check if the tracker is deterministic
                if r > 0 and tracker.is_deterministic:
                    break
                elif r == 3 and self._check_deterministic('baseline', tracker.name, seq_name):
                    ops.sys_print(
                        '  Detected a deterministic tracker, '
                        'skipping remaining trials.')
                    break
                ops.sys_print(' Repetition: %d' % (r + 1))

                # skip if results exist
                record_file = os.path.join(
                    self.result_dir, tracker.name, 'baseline', seq_name,
                    '%s_%03d.txt' % (seq_name, r + 1))
                if os.path.exists(record_file):
                    ops.sys_print('  Found results, skipping %s' % seq_name)
                    continue

                # state variables
                bboxes = []
                times = []
                failure = False
                next_start = -1

                # tracking loop
                for f, img_file in enumerate(img_files):
                    img = ops.read_image(img_file)
                    if tracker.input_type == 'image':
                        frame = img
                    elif tracker.input_type == 'file':
                        frame = img_file

                    start_time = time.time()
                    if f == 0:
                        # initial frame
                        tracker.init(frame, anno_rects[0])
                        bboxes.append([1])
                    elif failure:
                        # during failure frames
                        if f == next_start:
                            if np.all(anno_rects[f] <= 0):
                                next_start += 1
                                start_time = np.NaN
                                bboxes.append([0])
                            else:
                                failure = False
                                tracker.init(frame, anno_rects[f])
                                bboxes.append([1])
                        else:
                            start_time = np.NaN
                            bboxes.append([0])
                    else:
                        # during success frames
                        bbox = tracker.update(frame)
                        iou = ops.poly_iou(anno[f], bbox, bound=img.shape[1::-1])
                        if iou <= 0.0:
                            # tracking failure
                            failure = True
                            next_start = f + self.skip_initialize
                            bboxes.append([2])
                        else:
                            # tracking succeed
                            bboxes.append(bbox)
                    
                    # store elapsed time
                    times.append(time.time() - start_time)

                    # visualize if required
                    if visualize:
                        if len(bboxes[-1]) == 4:
                            ops.show_image(img, bboxes[-1])
                        else:
                            ops.show_image(img)
                
                # record results
                self._record(record_file, bboxes, times)

    def run_unsupervised(self, tracker, visualize=False):
        ops.sys_print('Running unsupervised experiment...')

        # loop over the complete dataset
        for s, (img_files, target) in enumerate(self.dataset):
            seq_name = self.dataset.seq_names[s]
            ops.sys_print('--Sequence %d/%d: %s' % (
                s + 1, len(self.dataset), seq_name))

            # skip if results exist
            record_file = os.path.join(
                self.result_dir, tracker.name, 'unsupervised', seq_name,
                '%s_001.txt' % seq_name)
            if os.path.exists(record_file):
                ops.sys_print('  Found results, skipping %s' % seq_name)
                continue

            # rectangular bounding boxes
            anno_rects = target['anno'].copy()
            if anno_rects.shape[1] == 8:
                anno_rects = self.dataset._corner2rect(anno_rects)

            # tracking loop
            bboxes, times = tracker.forward_test(
                img_files, anno_rects[0], visualize=visualize)
            assert len(bboxes) == len(img_files)

            # re-formatting
            bboxes = list(bboxes)
            bboxes[0] = [1]
            
            # record results
            self._record(record_file, bboxes, times)

    def run_realtime(self, tracker, visualize=False):
        ops.sys_print('Running real-time experiment...')

        # loop over the complete dataset
        for s, (img_files, target) in enumerate(self.dataset):
            seq_name = self.dataset.seq_names[s]
            ops.sys_print('--Sequence %d/%d: %s' % (
                s + 1, len(self.dataset), seq_name))

            # skip if results exist
            record_file = os.path.join(
                self.result_dir, tracker.name, 'realtime', seq_name,
                '%s_001.txt' % seq_name)
            if os.path.exists(record_file):
                ops.sys_print('  Found results, skipping %s' % seq_name)
                continue

            # rectangular bounding boxes
            anno = target['anno']
            anno_rects = anno.copy()
            if anno_rects.shape[1] == 8:
                anno_rects = self.dataset._corner2rect(anno_rects)

            # state variables
            bboxes = []
            times = []
            next_start = 0
            failure = False
            failed_frame = -1
            total_time = 0.0
            grace = 3 - 1
            offset = 0

            # tracking loop
            for f, img_file in enumerate(img_files):
                img = ops.read_image(img_file)
                if tracker.input_type == 'image':
                    frame = img
                elif tracker.input_type == 'file':
                    frame = img_file

                start_time = time.time()
                if f == next_start:
                    # during initial frames
                    tracker.init(frame, anno_rects[f])
                    bboxes.append([1])

                    # reset state variables
                    failure = False
                    failed_frame = -1
                    total_time = 0.0
                    grace = 3 - 1
                    offset = f
                elif not failure:
                    # during success frames
                    # calculate current frame
                    if grace > 0:
                        total_time += 1000.0 / 25
                        grace -= 1
                    else:
                        total_time += max(1000.0 / 25, last_time * 1000.0)
                    current = offset + int(np.round(np.floor(total_time * 25) / 1000.0))

                    # delayed/tracked bounding box
                    if f < current:
                        bbox = bboxes[-1]
                    elif f == current:
                        bbox = tracker.update(frame)

                    iou = ops.poly_iou(anno[f], bbox, bound=img.shape[1::-1])
                    if iou <= 0.0:
                        # tracking failure
                        failure = True
                        failed_frame = f
                        next_start = current + self.skip_initialize
                        bboxes.append([2])
                    else:
                        # tracking succeed
                        bboxes.append(bbox)
                else:
                    # during failure frames
                    if f < current:
                        # skipping frame due to slow speed
                        bboxes.append([0])
                        start_time = np.NaN
                    elif f == current:
                        # current frame
                        bbox = tracker.update(frame)
                        iou = ops.poly_iou(anno[f], bbox, bound=img.shape[1::-1])
                        if iou <= 0.0:
                            # tracking failure
                            bboxes.append([2])
                            bboxes[failed_frame] = [0]
                            times[failed_frame] = np.NaN
                        else:
                            # tracking succeed
                            bboxes.append(bbox)
                    elif f < next_start:
                        # skipping frame due to failure
                        bboxes.append([0])
                        start_time = np.NaN

                # store elapsed time
                last_time = time.time() - start_time
                times.append(last_time)

                # visualize if required
                if visualize:
                    if len(bboxes[-1]) == 4:
                        ops.show_image(img, bboxes[-1])
                    else:
                        ops.show_image(img)

            # record results
            self._record(record_file, bboxes, times)

    def report(self, tracker_names, plot_curves=False):
        if isinstance(tracker_names, str):
            tracker_names = [tracker_names]
        assert isinstance(tracker_names, (list, tuple))

        # function for loading results
        def read_record(filename):
            with open(filename) as f:
                record = f.read().strip().split('\n')
            record = [[float(t) for t in line.split(',')]
                      for line in record]
            return record

        # assume tracker_names[0] is your tracker
        report_dir = os.path.join(self.report_dir, tracker_names[0])
        if not os.path.exists(report_dir):
            os.makedirs(report_dir)
        report_file = os.path.join(report_dir, 'performance.json')

        performance = {}
        for name in tracker_names:
            ops.sys_print('Evaluating %s' % name)
            ious = {}
            ious_full = {}
            failures = {}
            times = {}
            masks = {}  # frame masks for attribute tags

            for s, (img_files, target) in enumerate(self.dataset):
                anno, meta = target['anno'], target['meta']
                seq_name = self.dataset.seq_names[s]

                # initialize frames scores
                frame_num = len(img_files)
                ious[seq_name] = np.full(
                    (self.repetitions, frame_num), np.nan, dtype=float)
                ious_full[seq_name] = np.full(
                    (self.repetitions, frame_num), np.nan, dtype=float)
                failures[seq_name] = np.full(
                    (self.repetitions, frame_num), np.nan, dtype=float)
                times[seq_name] = np.full(
                    (self.repetitions, frame_num), np.nan, dtype=float)

                # read results of all repetitions
                record_files = sorted(glob.glob(os.path.join(
                    self.result_dir, name, 'baseline', seq_name,
                    '%s_[0-9]*.txt' % seq_name)))
                bboxes = [read_record(f) for f in record_files]
                assert all([len(b) == len(anno) for b in bboxes])

                # calculate frame ious with burnin
                bound = ops.read_image(img_files[0]).shape[1::-1]
                seq_ious = [self._calc_iou(b, anno, bound, burnin=True)
                            for b in bboxes]
                ious[seq_name][:len(seq_ious), :] = seq_ious

                # calculate frame ious without burnin
                seq_ious_full = [self._calc_iou(b, anno, bound)
                                 for b in bboxes]
                ious_full[seq_name][:len(seq_ious_full), :] = seq_ious_full

                # calculate frame failures
                seq_failures = [
                    [len(b) == 1 and b[0] == 2 for b in bboxes_per_rep]
                    for bboxes_per_rep in bboxes]
                failures[seq_name][:len(seq_failures), :] = seq_failures

                # collect frame runtimes
                time_file = os.path.join(
                    self.result_dir, name, 'baseline', seq_name,
                    '%s_time.txt' % seq_name)
                if os.path.exists(time_file):
                    seq_times = np.loadtxt(time_file, delimiter=',').T
                    times[seq_name][:len(seq_times), :] = seq_times

                # collect attribute masks
                tag_num = len(self.tags)
                masks[seq_name] = np.zeros((tag_num, frame_num), bool)
                for i, tag in enumerate(self.tags):
                    if tag in meta:
                        masks[seq_name][i, :] = meta[tag]
                # frames with no tags
                if 'empty' in self.tags:
                    tag_frames = np.array([
                        v for k, v in meta.items()
                        if isinstance(v, np.ndarray) and \
                        not 'practical' in k], dtype=bool)
                    ind = self.tags.index('empty')
                    masks[seq_name][ind, :] = \
                        ~np.logical_or.reduce(tag_frames, axis=0)

            # concatenate frames
            seq_names = self.dataset.seq_names
            masks = np.concatenate(
                [masks[s] for s in seq_names], axis=1)
            ious = np.concatenate(
                [ious[s] for s in seq_names], axis=1)
            failures = np.concatenate(
                [failures[s] for s in seq_names], axis=1)

            with warnings.catch_warnings():
                # average over repetitions
                warnings.simplefilter('ignore', category=RuntimeWarning)
                ious = np.nanmean(ious, axis=0)
                failures = np.nanmean(failures, axis=0)
            
                # calculate average overlaps and failures for each tag
                tag_ious = np.array(
                    [np.nanmean(ious[m]) for m in masks])
                tag_failures = np.array(
                    [np.nansum(failures[m]) for m in masks])
                tag_frames = masks.sum(axis=1)

            # remove nan values
            tag_ious[np.isnan(tag_ious)] = 0.0
            tag_weights = tag_frames / tag_frames.sum()

            # calculate weighted accuracy and robustness
            accuracy = np.sum(tag_ious * tag_weights)
            robustness = np.sum(tag_failures * tag_weights)

            # calculate tracking speed
            times = np.concatenate([
                t.reshape(-1) for t in times.values()])
            # remove invalid values
            times = times[~np.isnan(times)]
            times = times[times > 0]
            if len(times) > 0:
                speed = np.mean(1. / times)
            else:
                speed = -1

            performance.update({name: {
                'accuracy': accuracy,
                'robustness': robustness,
                'speed_fps': speed}})

        # save performance
        with open(report_file, 'w') as f:
            json.dump(performance, f, indent=4)
        ops.sys_print('Performance saved at %s' % report_file)

        return performance

    def show(self, tracker_names, seq_names=None, play_speed=1,
             experiment='supervised', visualize=True,
             save=False, save_dir='screenshots'):
        if seq_names is None:
            seq_names = self.dataset.seq_names
        elif isinstance(seq_names, str):
            seq_names = [seq_names]
        assert isinstance(tracker_names, (list, tuple))
        assert isinstance(seq_names, (list, tuple))
        assert experiment in ['supervised', 'unsupervised', 'realtime']

        play_speed = int(round(play_speed))
        assert play_speed > 0

        # "supervised" experiment results are stored in "baseline" folder
        if experiment == 'supervised':
            experiment = 'baseline'

        # function for loading results
        def read_record(filename):
            with open(filename) as f:
                record = f.read().strip().split('\n')
            record = [[float(t) for t in line.split(',')]
                      for line in record]
            for i, r in enumerate(record):
                if len(r) == 4:
                    record[i] = np.array(r)
                elif len(r) == 8:
                    r = np.array(r)[np.newaxis, :]
                    r = self.dataset._corner2rect(r)
                    record[i] = r[0]
                else:
                    record[i] = np.zeros(4)
            return record

        for s, seq_name in enumerate(seq_names):
            ops.sys_print('[%d/%d] Showing results on %s...' % (
                s + 1, len(seq_names), seq_name))
            
            # mkdir if required to save screenshots
            if save:
                out_dir = os.path.join(save_dir, seq_name)
                if not os.path.exists(out_dir):
                    os.makedirs(out_dir)
            
            # load all tracking results
            records = {}
            for name in tracker_names:
                record_file = os.path.join(
                    self.result_dir, name, experiment, seq_name,
                    '%s_001.txt' % seq_name)
                records[name] = read_record(record_file)

            # loop over the sequence and display results
            img_files, target = self.dataset[seq_name][:2]
            anno = target['anno']
            if anno.shape[1] == 8:
                anno = self.dataset._corner2rect(anno)
            for f, img_file in enumerate(img_files):
                if not f % play_speed == 0:
                    continue
                img = ops.read_image(img_file)
                bboxes = [anno[f]] + [
                    records[name][f] for name in tracker_names]
                img = ops.show_image(img, bboxes, visualize=visualize)

                # save screenshot if required
                if save:
                    out_file = os.path.join(out_dir, '%08d.jpg' % (f + 1))
                    cv2.imwrite(out_file, img)

    def _record(self, record_file, bboxes, times):
        # convert bboxes to string
        lines = []
        for bbox in bboxes:
            if len(bbox) == 1:
                lines.append('%d' % bbox[0])
            else:
                lines.append(str.join(',', ['%.4f' % t for t in bbox]))

        # record bounding boxes
        record_dir = os.path.dirname(record_file)
        if not os.path.isdir(record_dir):
            os.makedirs(record_dir)
        with open(record_file, 'w') as f:
            f.write(str.join('\n', lines))
        ops.sys_print('  Results recorded at %s' % record_file)

        # convert times to string
        lines = ['%.4f' % t for t in times]
        lines = [t.replace('nan', 'NaN') for t in lines]

        # record running times
        time_file = record_file[:record_file.rfind('_')] + '_time.txt'
        if os.path.exists(time_file):
            with open(time_file) as f:
                exist_lines = f.read().strip().split('\n')
            lines = [t + ',' + s for t, s in zip(exist_lines, lines)]
        with open(time_file, 'w') as f:
            f.write(str.join('\n', lines))

    def _check_deterministic(self, exp, tracker_name, seq_name):
        record_dir = os.path.join(
            self.result_dir, tracker_name, exp, seq_name)
        record_files = sorted(glob.glob(os.path.join(
            record_dir, '%s_[0-9]*.txt' % seq_name)))

        if len(record_files) < 3:
            return False
        
        records = []
        for record_file in record_files:
            with open(record_file, 'r') as f:
                records.append(f.read())
        
        return len(set(records)) == 1

    def _calc_iou(self, bboxes, anno, bound, burnin=False):
        # skip initialization frames
        if burnin:
            bboxes = bboxes.copy()
            init_inds = [i for i, bbox in enumerate(bboxes)
                         if bbox == [1.0]]
            for ind in init_inds:
                bboxes[ind:ind + self.burnin] = [[0]] * self.burnin
        # calculate polygon ious
        ious = np.array([ops.poly_iou(np.array(a), b, bound)
                         if len(a) > 1 else np.NaN
                         for a, b in zip(bboxes, anno)])
        return ious


@registry.register_module
class EvaluatorVOT(VOT_Eval):

    def __init__(self, root_dir=None, version=2018, **kwargs):
        dataset = VOT(root_dir, version, anno_type='default')
        super(EvaluatorVOT, self).__init__(dataset, **kwargs)
