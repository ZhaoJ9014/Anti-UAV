import os
import os.path as osp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import json
import cv2

import libs.ops as ops
from libs.config import registry
from libs.data import datasets
from .evaluator import Evaluator


__all__ = ['OTB_Eval', 'EvaluatorOTB', 'EvaluatorDTB70', 'EvaluatorNfS',
           'EvaluatorTColor128', 'EvaluatorUAV123', 'EvaluatorLaSOT',
           'EvaluatorTLP', 'EvaluatorVisDrone']


@registry.register_module
class OTB_Eval(Evaluator):
    r"""Evaluation pipeline and evaluation toolkit for OTB-like datasets.

    Args:
        dataset (Dataset): An OTB-like dataset.
        nbins_iou (integer optional): Number of bins for plotting success curves
            and calculating success scores.
        nbins_ce (integer optional): Number of bins for plotting precision curves.
        result_dir (string, optional): Directory for storing tracking
            results. Default is ``./results``.
        report_dir (string, optional): Directory for storing performance
            evaluation results. Default is ``./reports``.
    """
    def __init__(self, dataset, nbins_iou=101, nbins_ce=51,
                 result_dir='results', report_dir='reports',
                 visualize=False, plot_curves=False, frame_stride=1):
        self.dataset = dataset
        self.result_dir = osp.join(result_dir, self.dataset.name)
        self.report_dir = osp.join(report_dir, self.dataset.name)
        if frame_stride != 1:
            self.result_dir += '_s{}'.format(frame_stride)
            self.report_dir += '_s{}'.format(frame_stride)
        self.visualize = visualize
        self.plot_curves = plot_curves
        self.frame_stride = frame_stride
        # as nbins_iou increases, the success score
        # converges to the average overlap (AO)
        self.nbins_iou = nbins_iou
        self.nbins_ce = 51

    def run(self, tracker, visualize=None):
        if visualize is None:
            visualize = self.visualize
        ops.sys_print('Running tracker %s on %s...' % (
            tracker.name, self.dataset.name))

        # loop over the complete dataset
        for s, (img_files, target) in enumerate(self.dataset):
            seq_name = self.dataset.seq_names[s]
            ops.sys_print('--Sequence %d/%d: %s' % (
                s + 1, len(self.dataset), seq_name))

            # skip if results exist
            record_file = osp.join(
                self.result_dir, tracker.name, '%s.txt' % seq_name)
            if osp.exists(record_file):
                ops.sys_print('  Found results, skipping', seq_name)
                continue

            # tracking loop
            img_files = img_files[::self.frame_stride]
            init_bbox = target['anno'][0]
            bboxes, times = tracker.forward_test(
                img_files, init_bbox, visualize=visualize)
            assert len(bboxes) == len(img_files)

            # record results
            self._record(record_file, bboxes, times)

    def report(self, tracker_names, plot_curves=None):
        if isinstance(tracker_names, str):
            tracker_names = [tracker_names]
        assert isinstance(tracker_names, (list, tuple))
        if plot_curves is None:
            plot_curves = self.plot_curves

        # assume that tracker_names[0] is your tracker
        report_dir = osp.join(self.report_dir, tracker_names[0])
        if not osp.isdir(report_dir):
            os.makedirs(report_dir)
        report_file = osp.join(report_dir, 'performance.json')

        performance = {}
        for name in tracker_names:
            ops.sys_print('Evaluating', name)
            seq_num = len(self.dataset)
            succ_curve = np.zeros((seq_num, self.nbins_iou))
            prec_curve = np.zeros((seq_num, self.nbins_ce))
            speeds = np.zeros(seq_num)

            performance.update({name: {
                'overall': {},
                'seq_wise': {}}})

            for s, (_, target) in enumerate(self.dataset):
                seq_name = self.dataset.seq_names[s]
                anno = target['anno'][::self.frame_stride]

                # load results
                record_file = osp.join(
                    self.result_dir, name, '%s.txt' % seq_name)
                bboxes = np.loadtxt(record_file, delimiter=',')
                bboxes[0] = anno[0]
                assert len(bboxes) == len(anno)

                # skip invalid frames
                mask = np.all(anno > 0, axis=1)
                assert mask.sum() > 0
                bboxes = bboxes[mask]
                anno = anno[mask]

                ious, center_errors = self._calc_metrics(bboxes, anno)
                succ_curve[s], prec_curve[s] = self._calc_curves(ious, center_errors)

                # calculate average tracking speed
                time_file = osp.join(
                    self.result_dir, name, 'times/%s_time.txt' % seq_name)
                if osp.isfile(time_file):
                    times = np.loadtxt(time_file)
                    times = times[times > 0]
                    if len(times) > 0:
                        speeds[s] = np.mean(1. / times)

                # store sequence-wise performance
                performance[name]['seq_wise'].update({seq_name: {
                    'success_curve': succ_curve[s].tolist(),
                    'precision_curve': prec_curve[s].tolist(),
                    'success_score': np.mean(succ_curve[s]),
                    'precision_score': prec_curve[s][20],
                    'success_rate': succ_curve[s][self.nbins_iou // 2],
                    'speed_fps': speeds[s] if speeds[s] > 0 else -1}})

            succ_curve = np.mean(succ_curve, axis=0)
            prec_curve = np.mean(prec_curve, axis=0)
            succ_score = np.mean(succ_curve)
            prec_score = prec_curve[20]
            succ_rate = succ_curve[self.nbins_iou // 2]
            if np.count_nonzero(speeds) > 0:
                avg_speed = np.sum(speeds) / np.count_nonzero(speeds)
            else:
                avg_speed = -1

            # store overall performance
            performance[name]['overall'].update({
                'success_curve': succ_curve.tolist(),
                'precision_curve': prec_curve.tolist(),
                'success_score': succ_score,
                'precision_score': prec_score,
                'success_rate': succ_rate,
                'speed_fps': avg_speed})

        # report the performance
        with open(report_file, 'w') as f:
            json.dump(performance, f, indent=4)

        if plot_curves:
            # plot precision and success curves
            self.plot_curves(tracker_names)

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
                    self.result_dir, name, '%s.txt' % seq_name)
                records[name] = np.loadtxt(record_file, delimiter=',')

            # loop over the sequence and display results
            img_files, target = self.dataset[seq_name][:2]
            img_files = img_files[::self.frame_stride]
            target['anno'] = target['anno'][self.frame_stride]
            for f, img_file in enumerate(img_files):
                if not f % play_speed == 0:
                    continue
                img = ops.read_image(img_file)
                bboxes = [records[name][f] for name in tracker_names]
                if len(target['anno']) > f:
                   bboxes = [target['anno'][f]] + bboxes
                img = ops.show_image(img, bboxes, visualize=visualize)

                # save screenshot if required
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
        time_dir = osp.join(record_dir, 'times')
        if not osp.isdir(time_dir):
            os.makedirs(time_dir)
        time_file = osp.join(time_dir, osp.basename(
            record_file).replace('.txt', '_time.txt'))
        np.savetxt(time_file, times, fmt='%.8f')

    def _calc_metrics(self, bboxes, anno):
        # can be modified by children classes
        ious = ops.rect_iou(bboxes, anno)
        center_errors = ops.center_error(bboxes, anno)
        return ious, center_errors

    def _calc_curves(self, ious, center_errors):
        ious = np.asarray(ious, float)[:, np.newaxis]
        center_errors = np.asarray(center_errors, float)[:, np.newaxis]

        thr_iou = np.linspace(0, 1, self.nbins_iou)[np.newaxis, :]
        thr_ce = np.arange(0, self.nbins_ce)[np.newaxis, :]

        bin_iou = np.greater(ious, thr_iou)
        bin_ce = np.less_equal(center_errors, thr_ce)

        succ_curve = np.mean(bin_iou, axis=0)
        prec_curve = np.mean(bin_ce, axis=0)

        return succ_curve, prec_curve

    def plot_curves(self, tracker_names):
        # assume tracker_names[0] is your tracker
        report_dir = osp.join(self.report_dir, tracker_names[0])
        assert osp.exists(report_dir), \
            'No reports found. Run "report" first ' \
            'before plotting curves.'
        report_file = osp.join(report_dir, 'performance.json')
        assert osp.exists(report_file), \
            'No reports found. Run "report" first ' \
            'before plotting curves.'

        # load pre-computed performance
        with open(report_file) as f:
            performance = json.load(f)

        succ_file = osp.join(report_dir, 'success_plots.pdf')
        prec_file = osp.join(report_dir, 'precision_plots.pdf')
        key = 'overall'

        # markers
        markers = ['-', '--', '-.']
        markers = [c + m for m in markers for c in [''] * 10]

        # sort trackers by success score
        tracker_names = list(performance.keys())
        succ = [t[key]['success_score'] for t in performance.values()]
        inds = np.argsort(succ)[::-1]
        tracker_names = [tracker_names[i] for i in inds]

        # plot success curves
        thr_iou = np.linspace(0, 1, self.nbins_iou)
        fig, ax = plt.subplots()
        lines = []
        legends = []
        for i, name in enumerate(tracker_names):
            line, = ax.plot(thr_iou,
                            performance[name][key]['success_curve'],
                            markers[i % len(markers)])
            lines.append(line)
            legends.append('%s: [%.3f]' % (
                name, performance[name][key]['success_score']))
        matplotlib.rcParams.update({'font.size': 7.4})
        legend = ax.legend(lines, legends, loc='center left',
                           bbox_to_anchor=(1, 0.5))

        matplotlib.rcParams.update({'font.size': 9})
        ax.set(xlabel='Overlap threshold',
               ylabel='Success rate',
               xlim=(0, 1), ylim=(0, 1),
               title='Success plots of OPE')
        ax.grid(True)
        fig.tight_layout()

        ops.sys_print('Saving success plots to', succ_file)
        fig.savefig(succ_file,
                    bbox_extra_artists=(legend,),
                    bbox_inches='tight',
                    dpi=300)

        # sort trackers by precision score
        tracker_names = list(performance.keys())
        prec = [t[key]['precision_score'] for t in performance.values()]
        inds = np.argsort(prec)[::-1]
        tracker_names = [tracker_names[i] for i in inds]

        # plot precision curves
        thr_ce = np.arange(0, self.nbins_ce)
        fig, ax = plt.subplots()
        lines = []
        legends = []
        for i, name in enumerate(tracker_names):
            line, = ax.plot(thr_ce,
                            performance[name][key]['precision_curve'],
                            markers[i % len(markers)])
            lines.append(line)
            legends.append('%s: [%.3f]' % (
                name, performance[name][key]['precision_score']))
        matplotlib.rcParams.update({'font.size': 7.4})
        legend = ax.legend(lines, legends, loc='center left',
                           bbox_to_anchor=(1, 0.5))

        matplotlib.rcParams.update({'font.size': 9})
        ax.set(xlabel='Location error threshold',
               ylabel='Precision',
               xlim=(0, thr_ce.max()), ylim=(0, 1),
               title='Precision plots of OPE')
        ax.grid(True)
        fig.tight_layout()

        ops.sys_print('Saving precision plots to', prec_file)
        fig.savefig(prec_file, dpi=300)


@registry.register_module
class EvaluatorOTB(OTB_Eval):

    def __init__(self, root_dir=None, version=2015, **kwargs):
        dataset = datasets.OTB(root_dir, version, download=True)
        if version == 2013:
            nbins_iou = 21
        else:
            nbins_iou = 101
        super(EvaluatorOTB, self).__init__(dataset, nbins_iou, **kwargs)


@registry.register_module
class EvaluatorDTB70(OTB_Eval):
    r"""Evaluation pipeline and evaluation toolkit for DTB70 dataset.

    Args:
        root_dir (string): Root directory of DTB70 dataset.
    """
    def __init__(self, root_dir=None, **kwargs):
        dataset = datasets.DTB70(root_dir)
        super(EvaluatorDTB70, self).__init__(dataset, **kwargs)


@registry.register_module
class EvaluatorNfS(OTB_Eval):
    r"""Evaluation pipeline and evaluation toolkit for NfS dataset.

    Args:
        root_dir (string): Root directory of NfS dataset.
        fps (integer): Choose version among 30 fps and 240 fps.
    """
    def __init__(self, root_dir=None, fps=30, **kwargs):
        dataset = datasets.NfS(root_dir, fps)
        super(EvaluatorNfS, self).__init__(dataset, **kwargs)


@registry.register_module
class EvaluatorTColor128(OTB_Eval):
    r"""Evaluation pipeline and evaluation toolkit for TColor128 dataset.

    Args:
        root_dir (string): Root directory of TColor128 dataset.
    """
    def __init__(self, root_dir=None, **kwargs):
        dataset = datasets.TColor128(root_dir)
        super(EvaluatorTColor128, self).__init__(dataset, **kwargs)


@registry.register_module
class EvaluatorUAV123(OTB_Eval):
    r"""Evaluation pipeline and evaluation toolkit for UAV123 dataset.

    Args:
        root_dir (string): Root directory of UAV123 dataset.
        version (string): Choose version among UAV123 and UAV20L.
    """
    def __init__(self, root_dir=None, version='UAV123', **kwargs):
        dataset = datasets.UAV123(root_dir, version)
        super(EvaluatorUAV123, self).__init__(dataset, **kwargs)

    def _calc_metrics(self, bboxes, anno):
        valid = ~np.any(np.isnan(anno), axis=1)
        if len(valid) == 0:
            ops.sys_print('Warning: no valid annotations')
            return None, None
        else:
            ious = ops.rect_iou(bboxes[valid, :], anno[valid, :])
            center_errors = ops.center_error(
                bboxes[valid, :], anno[valid, :])
            return ious, center_errors


@registry.register_module
class EvaluatorLaSOT(OTB_Eval):
    r"""Evaluation pipeline and evaluation toolkit for LaSOT dataset.

    Args:
        root_dir (string): Root directory of LaSOT dataset.
    """
    def __init__(self, root_dir=None, **kwargs):
        dataset = datasets.LaSOT(root_dir, subset='test')
        super(EvaluatorLaSOT, self).__init__(
            dataset, nbins_iou=21, **kwargs)


@registry.register_module
class EvaluatorTLP(OTB_Eval):
    r"""Evaluation pipeline and evaluation toolkit for TLP dataset.

    Args:
        root_dir (string): Root directory of TLP dataset.
    """
    def __init__(self, root_dir=None, **kwargs):
        dataset = datasets.TLP(root_dir)
        super(EvaluatorTLP, self).__init__(dataset, **kwargs)


@registry.register_module
class EvaluatorVisDrone(OTB_Eval):
    r"""Evaluation pipeline and evaluation toolkit for VisDrone dataset.

    Args:
        root_dir (string): Root directory of VisDrone dataset.
        subset (string, optional): Specify ``train`` or ``val``
            subset of VisDrone.
    """
    def __init__(self, root_dir=None, subset='val', **kwargs):
        dataset = datasets.VisDroneSOT(root_dir, subset=subset)
        super(EvaluatorVisDrone, self).__init__(dataset, **kwargs)
