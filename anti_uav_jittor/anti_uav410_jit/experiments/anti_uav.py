from __future__ import absolute_import, division, print_function

import os
import io
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import json
from PIL import Image

from datasets import AntiUAV410
from utils.metrics import rect_iou, center_error
from utils.viz import show_frame


class ExperimentAntiUAV410(object):
    r"""Experiment pipeline and evaluation toolkit for AntiUAV410 dataset.
    
    Args:
        root_dir (string): Root directory of AntiUAV410 dataset.
        result_dir (string, optional): Directory for storing tracking
            results. Default is ``./results``.
        report_dir (string, optional): Directory for storing performance
            evaluation results. Default is ``./reports``.
    """
    def __init__(self, root_dir, subset,
                 result_dir='results', report_dir='reports', start_idx=0, end_idx=None):
        super(ExperimentAntiUAV410, self).__init__()
        self.subset = subset
        self.dataset = AntiUAV410(os.path.join(root_dir, subset))
        self.result_dir = os.path.join(result_dir, 'AntiUAV410', subset)
        self.report_dir = os.path.join(report_dir, 'AntiUAV410', subset)
        # as nbins_iou increases, the success score
        # converges to the average overlap (AO)
        self.nbins_iou = 21
        self.nbins_ce = 51
        self.start_idx = start_idx
        self.end_idx = end_idx
        self.use_confs = True
        self.dump_as_csv = False
        self.att_name = ['Thermal Crossover', 'Out-of-View', 'Scale Variation',
                    'Fast Motion', 'Occlusion', 'Dynamic Background Clutter',
                    'Tiny Size', 'Small Size', 'Medium Size', 'Normal Size']
        self.att_fig_name = ['TC', 'OV', 'SV', 'FM', 'OC', 'DBC',
                        'TS', 'SS', 'MS', 'NS']

    def run(self, tracker, visualize=False):
        print('Running tracker %s on %s...' % (
            tracker.name, type(self.dataset).__name__))

        end_idx = self.end_idx
        if end_idx is None:
            end_idx = len(self.dataset)

        overall_performance = []

        for s in range(self.start_idx, end_idx):

            img_files, label_res = self.dataset[s]

            seq_name = self.dataset.seq_names[s]

            print('--Sequence %d/%d: %s' % (s + 1, len(self.dataset), seq_name))

            # skip if results exist
            record_file = os.path.join(
                self.result_dir, tracker.name, '%s.txt' % seq_name)

            if os.path.exists(record_file):
                print('  Found results, skipping', seq_name)
                continue

            # import pdb;pdb.set_trace()
            # tracking loop
            bboxes, times = tracker.forward_test(
                img_files, label_res['gt_rect'][0], visualize=visualize)

            # record results
            self._record(record_file, bboxes, times)
            SA_Score = self.eval(bboxes, label_res)
            overall_performance.append(SA_Score)
            print('%20s Fixed Measure: %.03f' % (seq_name, SA_Score))

        print('[Overall] Mixed Measure: %.03f\n' % (np.mean(overall_performance)))

    def iou(self, bbox1, bbox2):
        """
        Calculates the intersection-over-union of two bounding boxes.
        Args:
            bbox1 (numpy.array, list of floats): bounding box in format x,y,w,h.
            bbox2 (numpy.array, list of floats): bounding box in format x,y,w,h.
        Returns:
            int: intersection-over-onion of bbox1, bbox2
        """
        bbox1 = [float(x) for x in bbox1]
        bbox2 = [float(x) for x in bbox2]

        (x0_1, y0_1, w1_1, h1_1) = bbox1
        (x0_2, y0_2, w1_2, h1_2) = bbox2
        x1_1 = x0_1 + w1_1
        x1_2 = x0_2 + w1_2
        y1_1 = y0_1 + h1_1
        y1_2 = y0_2 + h1_2
        # get the overlap rectangle
        overlap_x0 = max(x0_1, x0_2)
        overlap_y0 = max(y0_1, y0_2)
        overlap_x1 = min(x1_1, x1_2)
        overlap_y1 = min(y1_1, y1_2)

        # check if there is an overlap
        if overlap_x1 - overlap_x0 <= 0 or overlap_y1 - overlap_y0 <= 0:
            return 0

        # if yes, calculate the ratio of the overlap to each ROI size and the unified size
        size_1 = (x1_1 - x0_1) * (y1_1 - y0_1)
        size_2 = (x1_2 - x0_2) * (y1_2 - y0_2)
        size_intersection = (overlap_x1 - overlap_x0) * (overlap_y1 - overlap_y0)
        size_union = size_1 + size_2 - size_intersection

        return size_intersection / size_union

    def not_exist(self, pred):

        if len(pred) == 1 or len(pred) == 0:
            return 1.0
        else:
            return 0.0

    def eval(self, out_res, label_res):

        measure_per_frame = []

        for _pred, _gt, _exist in zip(out_res, label_res['gt_rect'], label_res['exist']):

            if not _exist:
                measure_per_frame.append(self.not_exist(_pred))
            else:

                if len(_gt) < 4 or sum(_gt) == 0:
                    continue

                if len(_pred) == 4:
                    measure_per_frame.append(self.iou(_pred, _gt))
                else:
                    measure_per_frame.append(0.0)

                # try:
                #     measure_per_frame.append(iou(_pred, _gt))
                # except:
                #     measure_per_frame.append(0)

            # measure_per_frame.append(not_exist(_pred) if not _exist else iou(_pred, _gt))

        return np.mean(measure_per_frame)


    def report(self, trackers, plot_curves=True, plot_attcurves=True):

        assert isinstance(trackers, (list, tuple))

        if isinstance(trackers[0], dict):
            pass
        else:
            trackers = [
                {'name': trackers[0], 'path': os.path.join(
                self.result_dir, trackers[0]), 'mode': 1},
            ]

        # assume tracker_names[0] is your tracker
        report_dir = os.path.join(self.report_dir)
        if not os.path.isdir(report_dir):
            os.makedirs(report_dir)
        report_file = os.path.join(report_dir, 'performance.json')

        performance = {}
        tracker_names = []

        overall_SAs = []

        for trackerid, tracker in enumerate(trackers):

            name=tracker['name']
            mode=tracker['mode']
            tracker_names.append(name)
            print('Evaluating', name)
            seq_num = len(self.dataset)
            succ_curve = np.zeros((seq_num, self.nbins_iou))
            prec_curve = np.zeros((seq_num, self.nbins_ce))
            speeds = np.zeros(seq_num)

            performance.update({name: {
                'overall': {},
                'TC':{},
                'OV':{},
                'SV':{},
                'FM':{},
                'OC':{},
                'DBC':{},
                'TS':{},
                'SS':{},
                'MS':{},
                'NS':{},
                'seq_wise': {}}})

            overall_SA = []

            att_list = []

            for s, (_, label_res) in enumerate(self.dataset):
                seq_name = self.dataset.seq_names[s]
                record_file = os.path.join(
                    tracker['path'], '%s.txt' % seq_name)

                if name == 'SwinTrack-Tiny':
                    record_file = os.path.join(
                        tracker['path'], 'test_metrics/anti-uav410-%s/%s/bounding_box.txt' % (self.subset, seq_name))

                if name == 'SwinTrack-Base':
                    record_file = os.path.join(
                        tracker['path'], 'test_metrics/anti-uav410-%s/%s/bounding_box.txt' % (self.subset, seq_name))

                att_file = os.path.join(
                    'annos', self.subset, 'att', '%s.txt' % seq_name)
                with open(att_file, 'r') as f:
                    att_temp = np.loadtxt(io.StringIO(f.read().replace(',', ' ')))
                att_list.append(att_temp)

                try:
                    with open(record_file, 'r') as f:
                        boxestemp = json.load(f)['res']
                except:
                    with open(record_file, 'r') as f:
                        boxestemp = np.loadtxt(io.StringIO(f.read().replace(',', ' ')))

                if mode==2:
                    boxestemp[:, 2:] = boxestemp[:, 2:] - boxestemp[:, :2] + 1

                SA_Score = self.eval(boxestemp, label_res)
                overall_SA.append(SA_Score)

                boxes=[]
                for box in boxestemp:
                    if len(box)==4:
                        boxes.append(box)
                    else:
                        boxes.append([0, 0, 0, 0])
                boxes = np.array(boxes)

                anno=[]
                annotemp=label_res['gt_rect']

                for ann in annotemp:
                    if len(ann)==4:
                        anno.append(ann)
                    else:
                        anno.append([0,0,0,0])

                anno=np.array(anno)

                boxes[0] = anno[0]
                # boxes=np.around(boxes, decimals=4)
                if not (len(boxes) == len(anno)):
                    print('warning: %s anno do not match boxes' % seq_name)
                    len_min = min(len(boxes), len(anno))
                    boxes = boxes[:len_min]
                    anno = anno[:len_min]
                assert len(boxes) == len(anno)

                ious, center_errors = self._calc_metrics(boxes, anno)
                succ_curve[s], prec_curve[s] = self._calc_curves(ious, center_errors)

                # calculate average tracking speed
                time_file = os.path.join(
                    self.result_dir, name, 'times/%s_time.txt' % seq_name)
                if os.path.isfile(time_file):
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

            overall_SAs.append(np.mean(overall_SA))

            all_succ_curve = succ_curve.copy()
            all_prec_curve = prec_curve.copy()

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

            # store att performance
            att_array = np.array(att_list)

            for ii in range(len(self.att_name)):
                att_ids = np.where(att_array[:, ii] > 0)[0]

                if trackerid == 0:
                    self.att_name[ii] = self.att_name[ii]+'('+ str(len(att_ids))+')'

                # import pdb;pdb.set_trace()
                att_succ_curve = all_succ_curve[att_ids, :]
                att_prec_curve = all_prec_curve[att_ids, :]
                att_succ_curve = np.mean(att_succ_curve, axis=0)
                att_prec_curve = np.mean(att_prec_curve, axis=0)
                att_succ_score = np.mean(att_succ_curve)

                att_prec_score = att_prec_curve[20]
                # import pdb;pdb.set_trace()
                att_succ_rate = att_succ_curve[self.nbins_iou // 2]

                performance[name][self.att_fig_name[ii]].update({
                    'att_success_curve': att_succ_curve.tolist(),
                    'att_precision_curve': att_prec_curve.tolist(),
                    'att_success_score': att_succ_score,
                    'att_precision_score': att_prec_score,
                    'att_success_rate': att_succ_rate})

        # import pdb;pdb.set_trace()
        sa_file = os.path.join(report_dir, 'state_accuracy_scores.txt')
        if os.path.exists(sa_file):
            os.remove(sa_file)
        text = '[Overall] %20s %20s: %s' % ('Tracker name', 'Experiments metric', 'Scores')
        with open(sa_file, 'a', encoding='utf-8') as f:
            f.write(text)
            f.write('\n')
        print(text)

        for ii in range(len(overall_SAs)):
            text = '[Overall] %20s %20s: %.04f' % (trackers[ii]['name'], 'State accuracy', overall_SAs[ii])
            with open(sa_file, 'a', encoding='utf-8') as f:
                f.write(text)
                f.write('\n')
            print(text)

        print('Saving state accuracy scores to', sa_file)
        # report the performance
        with open(report_file, 'w') as f:
            json.dump(performance, f, indent=4)
        # plot precision and success curves
        if plot_curves:
            self.plot_curves(tracker_names)

        if plot_attcurves:

            for ii in range(len(self.att_name)):

                self.plot_attcurves(tracker_names, self.att_name[ii], self.att_fig_name[ii])


        return performance


    def show(self, tracker_names, seq_names=None, play_speed=1):
        if seq_names is None:
            seq_names = self.dataset.seq_names
        elif isinstance(seq_names, str):
            seq_names = [seq_names]
        assert isinstance(tracker_names, (list, tuple))
        assert isinstance(seq_names, (list, tuple))

        play_speed = int(round(play_speed))
        assert play_speed > 0

        for s, seq_name in enumerate(seq_names):
            print('[%d/%d] Showing results on %s...' % (
                s + 1, len(seq_names), seq_name))
            
            # load all tracking results
            records = {}
            for name in tracker_names:
                record_file = os.path.join(
                    self.result_dir, name, '%s.txt' % seq_name)
                records[name] = np.loadtxt(record_file, delimiter=',')
            
            # loop over the sequence and display results
            img_files, anno = self.dataset[seq_name]
            for f, img_file in enumerate(img_files):
                if not f % play_speed == 0:
                    continue
                image = Image.open(img_file)
                boxes = [anno[f]] + [
                    records[name][f] for name in tracker_names]
                show_frame(image, boxes,
                           legends=['GroundTruth'] + tracker_names,
                           colors=['w', 'r', 'g', 'b', 'c', 'm', 'y',
                                   'orange', 'purple', 'brown', 'pink'])

    def _record(self, record_file, boxes, times, confs=None):
        # record bounding boxes
        record_dir = os.path.dirname(record_file)
        if not os.path.isdir(record_dir):
            os.makedirs(record_dir)
        np.savetxt(record_file, boxes, fmt='%.3f', delimiter=',')
        while not os.path.exists(record_file):
            print('warning: recording failed, retrying...')
            np.savetxt(record_file, boxes, fmt='%.3f', delimiter=',')
        print('  Results recorded at', record_file)

        # record confidences (if available)
        if confs is not None:
            # convert confs to string
            lines = ['%.4f' % c for c in confs]
            lines[0] = ''

            conf_file = record_file.replace(".txt", "_confidence.value")
            with open(conf_file, 'w') as f:
                f.write(str.join('\n', lines))

        # record running times
        time_dir = os.path.join(record_dir, 'times')
        if not os.path.isdir(time_dir):
            os.makedirs(time_dir)
        time_file = os.path.join(time_dir, os.path.basename(
            record_file).replace('.txt', '_time.txt'))
        np.savetxt(time_file, times, fmt='%.8f')

    def _calc_metrics(self, boxes, anno):
        # can be modified by children classes
        ious = rect_iou(boxes, anno)
        center_errors = center_error(boxes, anno)
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
        report_dir = os.path.join(self.report_dir)
        assert os.path.exists(report_dir), \
            'No reports found. Run "report" first' \
            'before plotting curves.'
        report_file = os.path.join(report_dir, 'performance.json')
        assert os.path.exists(report_file), \
            'No reports found. Run "report" first' \
            'before plotting curves.'

        # load pre-computed performance
        with open(report_file) as f:
            performance = json.load(f)

        succ_file = os.path.join(report_dir, 'success_plots.pdf')
        prec_file = os.path.join(report_dir, 'precision_plots.pdf')
        key = 'overall'

        # markers
        markers = ['-', '--', '-.']
        markers = [c + m for m in markers for c in [''] * 10]

        # sort trackers by success score
        tracker_names = list(performance.keys())
        succ = [t[key]['success_score'] for t in performance.values()]
        inds = np.argsort(succ)[::-1]
        tracker_names = [tracker_names[i] for i in inds]

        # activate latex text rendering
        #matplotlib.rc('text', usetex=True)
        matplotlib.rcParams.update({'font.size': 6.8})

        # plot success curves
        thr_iou = np.linspace(0, 1, self.nbins_iou)
        # fig, ax = plt.subplots(1, 1, figsize=(6, 4))
        fig, ax = plt.subplots()
        lines = []
        legends = []
        for i, name in enumerate(tracker_names):
            line, = ax.plot(thr_iou,
                            performance[name][key]['success_curve'],
                            markers[i % len(markers)],
                            linewidth=2,
                            zorder=performance[name][key]['success_score'])
            lines.append(line)
            if name == "Siam R-CNN":
                legends.append('$\\bf{Siam}$ $\\bf{R}$$\\bf{-}$$\\bf{CNN}$: [%.3f]' % performance[name][key]['success_score'])
            else:
                legends.append('%s: [%.3f]' % (name, performance[name][key]['success_score']))
        # matplotlib.rcParams.update({'font.size': 15.0})

        # for Default trackers
        legend = ax.legend(lines, legends, bbox_to_anchor=(0.98,-0.19), loc="lower right",
                bbox_transform=fig.transFigure, ncol=4, frameon=False)

        # for Re-trained trackers
        # legend = ax.legend(lines, legends, bbox_to_anchor=(0.98, -0.06), loc="lower right",
        #                    bbox_transform=fig.transFigure, ncol=4, frameon=False)
        
        matplotlib.rcParams.update({'font.size': 9.0})

        #matplotlib.rcParams.update({'font.size': 11})
        ax.set(xlabel='Overlap threshold',
               ylabel='Success rate',
               xlim=(0, 1), ylim=(0, 1.0),
               title='Success plots of OPE')
        ax.set_title('Success plots of OPE', fontweight='bold')
        # ax.xaxis.label.set_size(17)
        # ax.yaxis.label.set_size(17)
        ax.grid(True)
        fig.tight_layout()

        print('Saving success plots to', succ_file)
        fig.savefig(succ_file,
                    bbox_extra_artists=(legend,),
                    bbox_inches='tight',
                    dpi=300)
        # plt.show()
        # sort trackers by precision score
        tracker_names = list(performance.keys())
        prec = [t[key]['precision_score'] for t in performance.values()]
        # modified by Paul: instead use sorting from before so that colors of both plots are consistent
        inds = np.argsort(prec)[::-1]
        tracker_names = [tracker_names[i] for i in inds]

        # plot precision curves
        matplotlib.rcParams.update({'font.size': 6.8})
        thr_ce = np.arange(0, self.nbins_ce)
        fig, ax = plt.subplots()
        lines = []
        legends = []
        for i, name in enumerate(tracker_names):
            line, = ax.plot(thr_ce,
                            performance[name][key]['precision_curve'],
                            markers[i % len(markers)],
                            linewidth=2,
                            zorder=performance[name][key]['precision_score'])
            lines.append(line)
            if name == "Siam R-CNN":
                legends.append('$\\bf{Siam}$ $\\bf{R}$$\\bf{-}$$\\bf{CNN}$: [%.3f]' % performance[name][key]['precision_score'])
            else:
                legends.append('%s: [%.3f]' % (name, performance[name][key]['precision_score']))
        # matplotlib.rcParams.update({'font.size': 7.4})

        # for Default trackers
        legend = ax.legend(lines, legends, bbox_to_anchor=(0.97,-0.19), loc="lower right",
                bbox_transform=fig.transFigure, ncol=4, frameon=False)

        # for Re-trained trackers
        # legend = ax.legend(lines, legends, bbox_to_anchor=(0.97, -0.06), loc="lower right",
        #                    bbox_transform=fig.transFigure, ncol=4, frameon=False)

        matplotlib.rcParams.update({'font.size': 9.0})

        # matplotlib.rcParams.update({'font.size': 11})
        ax.set(xlabel='Location error threshold',
               ylabel='Precision',
               xlim=(0, thr_ce.max()), ylim=(0, 1.0),
               title='Precision plots of OPE')
        ax.set_title('Precision plots of OPE', fontweight='bold')
        # ax.xaxis.label.set_size(17)
        # ax.yaxis.label.set_size(17)
        ax.grid(True)
        fig.tight_layout()

        print('Saving precision plots to', prec_file)
        fig.savefig(prec_file,
                    bbox_extra_artists=(legend,),
                    bbox_inches='tight',
                    dpi=300)

    def plot_attcurves(self, tracker_names, att_name, att_key):
        # assume tracker_names[0] is your tracker
        report_dir = os.path.join(self.report_dir)
        assert os.path.exists(report_dir), \
            'No reports found. Run "report" first' \
            'before plotting curves.'
        report_file = os.path.join(report_dir, 'performance.json')
        assert os.path.exists(report_file), \
            'No reports found. Run "report" first' \
            'before plotting curves.'

        # load pre-computed performance
        with open(report_file) as f:
            performance = json.load(f)

        succ_file = os.path.join(report_dir, 'success_plots_of_'+att_key+'.pdf')
        prec_file = os.path.join(report_dir, 'precision_plots_of_'+att_key+'.pdf')

        # markers
        markers = ['-', '--', '-.']
        markers = [c + m for m in markers for c in [''] * 10]

        # sort trackers by success score
        tracker_names = list(performance.keys())
        succ = [t[att_key]['att_success_score'] for t in performance.values()]
        inds = np.argsort(succ)[::-1]
        tracker_names = [tracker_names[i] for i in inds]

        # activate latex text rendering
        # matplotlib.rc('text', usetex=True)
        matplotlib.rcParams.update({'font.size': 6.8})

        # plot success curves
        thr_iou = np.linspace(0, 1, self.nbins_iou)
        # fig, ax = plt.subplots(1, 1, figsize=(6, 4))
        fig, ax = plt.subplots()
        lines = []
        legends = []
        for i, name in enumerate(tracker_names):
            line, = ax.plot(thr_iou,
                            performance[name][att_key]['att_success_curve'],
                            markers[i % len(markers)],
                            linewidth=2,
                            zorder=performance[name][att_key]['att_success_score'])
            lines.append(line)
            if name == "Siam R-CNN":
                legends.append(
                    '$\\bf{Siam}$ $\\bf{R}$$\\bf{-}$$\\bf{CNN}$: [%.3f]' % performance[name][att_key]['att_success_score'])
            else:
                legends.append('%s: [%.3f]' % (name, performance[name][att_key]['att_success_score']))
        # matplotlib.rcParams.update({'font.size': 15.0})

        # for Default trackers
        legend = ax.legend(lines, legends, bbox_to_anchor=(0.98, -0.19), loc="lower right",
                           bbox_transform=fig.transFigure, ncol=4, frameon=False)

        # for Re-trained trackers
        # legend = ax.legend(lines, legends, bbox_to_anchor=(0.98, -0.06), loc="lower right",
        #                    bbox_transform=fig.transFigure, ncol=4, frameon=False)

        matplotlib.rcParams.update({'font.size': 9.0})

        # matplotlib.rcParams.update({'font.size': 11})
        ax.set(xlabel='Overlap threshold',
               ylabel='Success rate',
               xlim=(0, 1), ylim=(0, 1.0),
               title='Success plots of OPE - ' + att_name)
        ax.set_title('Success plots of OPE - ' + att_name, fontweight='bold')
        # ax.xaxis.label.set_size(17)
        # ax.yaxis.label.set_size(17)
        ax.grid(True)
        fig.tight_layout()

        print('Saving success plots to', succ_file)
        fig.savefig(succ_file,
                    bbox_extra_artists=(legend,),
                    bbox_inches='tight',
                    dpi=300)
        # plt.show()
        # sort trackers by precision score
        tracker_names = list(performance.keys())
        prec = [t[att_key]['att_precision_score'] for t in performance.values()]
        # modified by Paul: instead use sorting from before so that colors of both plots are consistent
        inds = np.argsort(prec)[::-1]
        tracker_names = [tracker_names[i] for i in inds]

        # plot precision curves
        matplotlib.rcParams.update({'font.size': 6.8})
        thr_ce = np.arange(0, self.nbins_ce)
        fig, ax = plt.subplots()
        lines = []
        legends = []
        for i, name in enumerate(tracker_names):
            line, = ax.plot(thr_ce,
                            performance[name][att_key]['att_precision_curve'],
                            markers[i % len(markers)],
                            linewidth=2,
                            zorder=performance[name][att_key]['att_precision_score'])
            lines.append(line)
            if name == "Siam R-CNN":
                legends.append('$\\bf{Siam}$ $\\bf{R}$$\\bf{-}$$\\bf{CNN}$: [%.3f]' % performance[name][att_key][
                    'att_precision_score'])
            else:
                legends.append('%s: [%.3f]' % (name, performance[name][att_key]['att_precision_score']))
        # matplotlib.rcParams.update({'font.size': 7.4})

        # for Default trackers
        legend = ax.legend(lines, legends, bbox_to_anchor=(0.97, -0.19), loc="lower right",
                           bbox_transform=fig.transFigure, ncol=4, frameon=False)

        # for Re-trained trackers
        # legend = ax.legend(lines, legends, bbox_to_anchor=(0.97, -0.06), loc="lower right",
        #                    bbox_transform=fig.transFigure, ncol=4, frameon=False)

        matplotlib.rcParams.update({'font.size': 9.0})

        # matplotlib.rcParams.update({'font.size': 11})
        ax.set(xlabel='Location error threshold',
               ylabel='Precision',
               xlim=(0, thr_ce.max()), ylim=(0, 1.0),
               title='Precision plots of OPE - '+ att_name)
        ax.set_title('Precision plots of OPE - '+ att_name, fontweight='bold')
        # ax.xaxis.label.set_size(17)
        # ax.yaxis.label.set_size(17)
        ax.grid(True)
        fig.tight_layout()

        print('Saving precision plots to', prec_file)
        fig.savefig(prec_file,
                    bbox_extra_artists=(legend,),
                    bbox_inches='tight',
                    dpi=300)
