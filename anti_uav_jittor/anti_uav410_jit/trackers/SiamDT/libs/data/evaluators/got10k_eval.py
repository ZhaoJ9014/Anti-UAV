import os
import os.path as osp
import numpy as np
import glob
import ast
import json
import time
import matplotlib.pyplot as plt
import matplotlib

import libs.ops as ops
from libs.config import registry
from libs.data.datasets import GOT10k
from .evaluator import Evaluator


__all__ = ['GOT10kEval', 'EvaluatorGOT10k']


@registry.register_module
class GOT10kEval(Evaluator):
    r"""Evaluation pipeline and evaluation toolkit for GOT-10k dataset.

    Args:
        root_dir (string): Root directory of GOT-10k dataset where
            ``train``, ``val`` and ``test`` folders exist.
        subset (string): Specify ``train``, ``val`` or ``test``
            subset of GOT-10k.
        list_file (string, optional): If provided, only run evaluation on
            sequences specified by this file.
        result_dir (string, optional): Directory for storing tracking
            results. Default is ``./results``.
        report_dir (string, optional): Directory for storing performance
            evaluation results. Default is ``./reports``.
    """
    def __init__(self, dataset, nbins_iou=101, repetitions=3,
                 result_dir='results', report_dir='reports'):
        self.dataset = dataset
        self.nbins_iou = nbins_iou
        self.repetitions = repetitions
        self.result_dir = osp.join(result_dir, self.dataset.name)
        self.report_dir = osp.join(report_dir, self.dataset.name)

    def run(self, tracker, visualize=False):
        if self.dataset.subset == 'test':
            ops.sys_print(
                '\033[93m[WARNING]:\n' \
                'The groundtruths of GOT-10k\'s test set is withholded.\n' \
                'You will have to submit your results to\n' \
                '[http://got-10k.aitestunion.com/]' \
                '\nto access the performance.\033[0m')
            time.sleep(2)
        ops.sys_print('Running tracker %s on GOT-10k...' % tracker.name)

        # loop over the complete dataset
        for s, (img_files, target) in enumerate(self.dataset):
            seq_name = self.dataset.seq_names[s]
            ops.sys_print('--Sequence %d/%d: %s' % (
                s + 1, len(self.dataset), seq_name))

            # run multiple repetitions for each sequence
            for r in range(self.repetitions):
                # check if the tracker is deterministic
                if r > 0 and tracker.is_deterministic:
                    break
                elif r == 3 and self._check_deterministic(
                    tracker.name, seq_name):
                    ops.sys_print('  Detected a deterministic tracker, ' +
                                  'skipping remaining trials.')
                    break
                ops.sys_print(' Repetition: %d' % (r + 1))

                # skip if results exist
                record_file = osp.join(
                    self.result_dir, tracker.name, seq_name,
                    '%s_%03d.txt' % (seq_name, r + 1))
                if osp.exists(record_file):
                    ops.sys_print('  Found results, skipping %s' % seq_name)
                    continue

                # tracking loop
                bboxes, times = tracker.forward_test(
                    img_files, target['anno'][0], visualize=visualize)

                # 需要转换一下
                bboxes[:, 2:] = bboxes[:, 2:] - bboxes[:, :2] + 1

                # record results
                self._record(record_file, bboxes, times)

    def report(self, tracker_names, plot_curves=False):
        if isinstance(tracker_names, str):
            tracker_names = [tracker_names]
        assert isinstance(tracker_names, (list, tuple))

        if self.dataset.subset == 'test':
            pwd = os.getcwd()

            # generate compressed submission file for each tracker
            for tracker_name in tracker_names:
                # compress all tracking results
                result_dir = osp.join(self.result_dir, tracker_name)
                os.chdir(result_dir)
                save_file = '../%s' % tracker_name
                ops.compress('.', save_file)
                ops.sys_print('Records saved at %s' % (save_file + '.zip'))

            # print submission guides
            ops.sys_print('\033[93mLogin and follow instructions on')
            ops.sys_print('http://got-10k.aitestunion.com/submit_instructions')
            ops.sys_print('to upload and evaluate your tracking results\033[0m')

            # switch back to previous working directory
            os.chdir(pwd)

            return None
        elif self.dataset.subset == 'val':
            # assume tracker_names[0] is your tracker
            report_dir = osp.join(self.report_dir, tracker_names[0])
            if not osp.exists(report_dir):
                os.makedirs(report_dir)
            report_file = osp.join(report_dir, 'performance.json')

            # visible ratios of all sequences
            seq_names = self.dataset.seq_names
            covers = {s: self.dataset[s][1]['meta']['cover'][1:]
                      for s in seq_names}

            performance = {}
            for name in tracker_names:
                ops.sys_print('Evaluating %s' % name)
                ious = {}
                times = {}
                performance.update({name: {
                    'overall': {},
                    'seq_wise': {}}})

                for s, (_, target) in enumerate(self.dataset):
                    seq_name = self.dataset.seq_names[s]
                    anno, meta = target['anno'], target['meta']

                    record_files = glob.glob(osp.join(
                        self.result_dir, name, seq_name,
                        '%s_[0-9]*.txt' % seq_name))
                    if len(record_files) == 0:
                        raise Exception('Results for sequence %s not found.' % seq_name)

                    # read results of all repetitions
                    bboxes = [np.loadtxt(f, delimiter=',') for f in record_files]
                    assert all([b.shape == anno.shape for b in bboxes])

                    # calculate and stack all ious
                    bound = ast.literal_eval(meta['resolution'])
                    seq_ious = [ops.rect_iou(
                        b[1:], anno[1:], bound=bound) for b in bboxes]
                    # only consider valid frames where targets are visible
                    seq_ious = [t[covers[seq_name] > 0] for t in seq_ious]
                    seq_ious = np.concatenate(seq_ious)
                    ious[seq_name] = seq_ious

                    # stack all tracking times
                    times[seq_name] = []
                    time_file = osp.join(
                        self.result_dir, name, seq_name,
                        '%s_time.txt' % seq_name)
                    if osp.exists(time_file):
                        seq_times = np.loadtxt(time_file, delimiter=',')
                        seq_times = seq_times[~np.isnan(seq_times)]
                        seq_times = seq_times[seq_times > 0]
                        if len(seq_times) > 0:
                            times[seq_name] = seq_times

                    # store sequence-wise performance
                    ao, sr, speed, _ = self._evaluate(seq_ious, seq_times)
                    performance[name]['seq_wise'].update({seq_name: {
                        'ao': ao,
                        'sr': sr,
                        'speed_fps': speed,
                        'length': len(anno) - 1}})

                ious = np.concatenate(list(ious.values()))
                times = np.concatenate(list(times.values()))

                # store overall performance
                ao, sr, speed, succ_curve = self._evaluate(ious, times)
                performance[name].update({'overall': {
                    'ao': ao,
                    'sr': sr,
                    'speed_fps': speed,
                    'succ_curve': succ_curve.tolist()}})

            # save performance
            with open(report_file, 'w') as f:
                json.dump(performance, f, indent=4)
            if plot_curves:
                # plot success curves
                self.plot_curves([report_file], tracker_names)

            return performance

    def show(self, tracker_names, seq_names=None, play_speed=1,
             visualize=True, save=False, save_dir='screenshots'):
        if seq_names is None:
            seq_names = self.dataset.seq_names
        elif isinstance(seq_names, str):
            seq_names = [seq_names]
        assert isinstance(tracker_names, (list, tuple))
        assert isinstance(seq_names, (list, tuple))

        play_speed = int(round(play_speed))
        assert play_speed > 0

        for s, seq_name in enumerate(seq_names):
            ops.sys_print('[%d/%d] Showing results on %s...' % (
                s + 1, len(seq_names), seq_name))

            # mkdir if required to save screenshots
            if save:
                out_dir = osp.join(save_dir, seq_name)
                if not osp.exists(out_dir):
                    os.makedirs(out_dir)

            # load all tracking results
            records = {}
            for name in tracker_names:
                record_file = osp.join(
                    self.result_dir, name, seq_name,
                    '%s_001.txt' % seq_name)
                records[name] = np.loadtxt(record_file, delimiter=',')

            # loop over the sequence and display results
            img_files, target = self.dataset[seq_name]
            for f, img_file in enumerate(img_files):
                if not f % play_speed == 0:
                    continue
                img = ops.read_image(img_file)
                bboxes = [target['anno'][f]] + [
                    records[name][f] for name in tracker_names]
                img = ops.show_image(img, bboxes, visualize=visualize)

                # save screenshots if required
                if save:
                    out_file = osp.join(out_dir, '%08d.jpg' % (f + 1))
                    cv2.imwrite(out_file, img)

    def _record(self, record_file, bboxes, times):
        # record bounding boxes
        record_dir = osp.dirname(record_file)
        if not osp.isdir(record_dir):
            os.makedirs(record_dir)
        np.savetxt(record_file, bboxes, fmt='%.3f', delimiter=',')
        ops.sys_print('  Results recorded at %s' % record_file)

        # record running times
        time_file = record_file[:record_file.rfind('_')] + '_time.txt'
        times = times[:, np.newaxis]
        if osp.exists(time_file):
            exist_times = np.loadtxt(time_file, delimiter=',')
            if exist_times.ndim == 1:
                exist_times = exist_times[:, np.newaxis]
            times = np.concatenate((exist_times, times), axis=1)
        np.savetxt(time_file, times, fmt='%.8f', delimiter=',')

    def _check_deterministic(self, tracker_name, seq_name):
        record_dir = osp.join(
            self.result_dir, tracker_name, seq_name)
        record_files = sorted(glob.glob(osp.join(
            record_dir, '%s_[0-9]*.txt' % seq_name)))

        if len(record_files) < 3:
            return False

        records = []
        for record_file in record_files:
            with open(record_file, 'r') as f:
                records.append(f.read())

        return len(set(records)) == 1

    def _evaluate(self, ious, times):
        # AO, SR and tracking speed
        ao = np.mean(ious)
        sr = np.mean(ious > 0.5)
        if len(times) > 0:
            # times has to be an array of positive values
            speed_fps = np.mean(1. / times)
        else:
            speed_fps = -1

        # success curve
        thr_iou = np.linspace(0, 1, 101)
        bin_iou = np.greater(ious[:, None], thr_iou[None, :])
        succ_curve = np.mean(bin_iou, axis=0)

        return ao, sr, speed_fps, succ_curve

    def plot_curves(self, report_files, tracker_names):
        assert isinstance(report_files, list), \
            'Expected "report_files" to be a list, ' \
            'but got %s instead' % type(report_files)

        # assume tracker_names[0] is your tracker
        report_dir = osp.join(self.report_dir, tracker_names[0])
        if not osp.exists(report_dir):
            os.makedirs(report_dir)

        performance = {}
        for report_file in report_files:
            with open(report_file) as f:
                performance.update(json.load(f))

        succ_file = osp.join(report_dir, 'success_plot.pdf')
        key = 'overall'

        # sort trackers by AO
        tracker_names = list(performance.keys())
        aos = [t[key]['ao'] for t in performance.values()]
        inds = np.argsort(aos)[::-1]
        tracker_names = [tracker_names[i] for i in inds]

        # markers
        markers = ['-', '--', '-.']
        markers = [c + m for m in markers for c in [''] * 10]

        # plot success curves
        thr_iou = np.linspace(0, 1, self.nbins_iou)
        fig, ax = plt.subplots()
        lines = []
        legends = []
        for i, name in enumerate(tracker_names):
            line, = ax.plot(thr_iou,
                            performance[name][key]['succ_curve'],
                            markers[i % len(markers)])
            lines.append(line)
            legends.append('%s: [%.3f]' % (
                name, performance[name][key]['ao']))
        matplotlib.rcParams.update({'font.size': 7.4})
        legend = ax.legend(lines, legends, loc='center left',
                           bbox_to_anchor=(1, 0.5))

        matplotlib.rcParams.update({'font.size': 9})
        ax.set(xlabel='Overlap threshold',
               ylabel='Success rate',
               xlim=(0, 1), ylim=(0, 1),
               title='Success plots on GOT-10k')
        ax.grid(True)
        fig.tight_layout()

        ops.sys_print('Saving success plots to %s' % succ_file)
        fig.savefig(succ_file,
                    bbox_extra_artists=(legend,),
                    bbox_inches='tight',
                    dpi=300)


@registry.register_module
class EvaluatorGOT10k(GOT10kEval):
    r"""Evaluation pipeline and evaluation toolkit for GOT-10k dataset.

    Args:
        root_dir (string): Root directory of GOT-10k dataset where
            ``train``, ``val`` and ``test`` folders exist.
        subset (string): Specify ``train``, ``val`` or ``test``
            subset of GOT-10k.
        list_file (string, optional): If provided, only run evaluation on
            sequences specified by this file.
    """
    def __init__(self, root_dir=None, subset='val',
                 list_file=None, **kwargs):
        assert subset in ['val', 'test']
        dataset = GOT10k(root_dir, subset=subset, list_file=list_file)
        super(EvaluatorGOT10k, self).__init__(dataset, **kwargs)
