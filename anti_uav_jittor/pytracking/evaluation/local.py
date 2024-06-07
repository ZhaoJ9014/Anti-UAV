from pytracking.evaluation.environment import EnvSettings

def local_env_settings():
    settings = EnvSettings()

    # Set your local paths here.

    settings.davis_dir = ''
    settings.got10k_path = '/home/cx/toolkit/got10k/datasets/'
    settings.got_packed_results_path = '/home/cx/TransT/pytracking/tracking_results/got_packed_results/'
    settings.got_reports_path = ''
    settings.lasot_path = '/home/cx/cx3/LaSOTBenchmark'
    settings.network_path = '/home/cx/detr-tracking/pytracking/networks/'    # Where tracking networks are stored.
    settings.nfs_path = '/home/cx/cx2/Downloads/nfs'
    settings.otb_path = '/home/cx/cx2/OTB100'
    settings.result_plot_path = '/home/cx/TransT/pytracking/result_plots/'
    settings.results_path = '/home/cx/TransT/pytracking/tracking_results'    # Where to store tracking results
    settings.segmentation_path = '/home/cx/TransT/pytracking/segmentation_results/'
    settings.tn_packed_results_path = ''
    settings.tpl_path = ''
    settings.trackingnet_path = ''
    settings.uav_path = '/home/cx/cx2/Downloads/UAV123/UAV123_fix/Dataset_UAV123/UAV123'
    settings.vot_path = '/home/cx/cx3/VOT2018'
    settings.youtubevos_dir = ''

    return settings

