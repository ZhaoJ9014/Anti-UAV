import os
# loss function related
from lib.utils.box_ops import ciou_loss, edl_mse_loss
# from torch.nn.functional import l1_loss
import jittor as jt
# from torch.nn import BCEWithLogitsLoss
# train pipeline related
from lib.train.trainers import LTRTrainer
# distributed training related
# from torch.nn.parallel import DistributedDataParallel as DDP
# some more advanced functions
from .base_functions import *
# network related
from lib.models.uavtrack import build_uavtrack, build_uavtrack_eh
# forward propagation related
from lib.train.actors import UAVTrackActor
# for import modules
import importlib


def prepare_input(res):
    res_t, res_s = res
    t = jt.Var(1, 3, res_t, res_t).cuda()
    s = t.Var(1, 3, res_s, res_s).cuda()
    return dict(template=t, search=s)


def run(settings):
    settings.description = 'Training script for UAV Tracking branch'

    # update the default configs with config file
    if not os.path.exists(settings.cfg_file):
        raise ValueError("%s doesn't exist." % settings.cfg_file)
    config_module = importlib.import_module("lib.config.%s.config" % settings.script_name)
    cfg = config_module.cfg
    config_module.update_config_from_file(settings.cfg_file)
    if settings.local_rank in [-1, 0]:
        print("New configuration is shown below.")
        for key in cfg.keys():
            print("%s configuration:" % key, cfg[key])
            print('\n')

    # update settings based on cfg
    update_settings(settings, cfg)

    # Record the training log
    log_dir = os.path.join(settings.save_dir, 'logs')
    if settings.local_rank in [-1, 0]:
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
    settings.log_file = os.path.join(log_dir, "%s-%s.log" % (settings.script_name, settings.config_name))

    # Build dataloaders
    loader_train, loader_val = build_dataloaders(cfg, settings)

    # Create network
    if settings.script_name == "uavtrack":
        net = build_uavtrack(cfg)
    elif settings.script_name == "uavtrack_eh":
        net = build_uavtrack_eh(cfg, settings)
    else:
        raise ValueError("illegal script name")

    # wrap networks to distributed one
    net.cuda()
    # if settings.local_rank != -1:
    #     # net = DDP(net, device_ids=[settings.local_rank], find_unused_parameters=True)
    #     # settings.device = torch.device("cuda:%d" % settings.local_rank)
    #     pass
    # else:
    #     # settings.device = torch.device("cuda:0")
    settings.deep_sup = getattr(cfg.TRAIN, "DEEP_SUPERVISION", False)
    settings.distill = getattr(cfg.TRAIN, "DISTILL", False)
    settings.distill_loss_type = getattr(cfg.TRAIN, "DISTILL_LOSS_TYPE", "KL")
    # settings.save_every_epoch = True
    # Loss functions and Actors
    if settings.script_name == "uavtrack":
        objective = {'ciou': ciou_loss, 'l1': jt.nn.l1_loss}
        loss_weight = {'ciou': cfg.TRAIN.IOU_WEIGHT, 'l1': cfg.TRAIN.L1_WEIGHT}
        actor = UAVTrackActor(net=net, objective=objective, loss_weight=loss_weight, settings=settings)
    elif settings.script_name == "uavtrack_eh":
        objective = {'ciou': ciou_loss, 'l1': jt.nn.l1_loss, 'score': edl_mse_loss if cfg.TRAIN.USE_UNCERTAINTY else jt.nn.BCEWithLogitsLoss()}
        loss_weight = {'ciou': cfg.TRAIN.IOU_WEIGHT, 'l1': cfg.TRAIN.L1_WEIGHT, 'score': cfg.TRAIN.SCORE_WEIGHT}
        actor = UAVTrackActor(net=net, objective=objective, loss_weight=loss_weight, settings=settings, run_score_head=True)
    else:
        raise ValueError("illegal script name")

    # Optimizer, parameters, and learning rates
    optimizer, lr_scheduler = get_optimizer_scheduler(net, cfg)
    use_amp = getattr(cfg.TRAIN, "AMP", False)
    accum_iter = getattr(cfg.TRAIN, "ACCUM_ITER", 1)
    trainer = LTRTrainer(actor, [loader_train, loader_val], optimizer, settings, lr_scheduler, accum_iter=accum_iter, use_amp=use_amp)

    # train process
    trainer.train(cfg.TRAIN.EPOCH, load_latest=True, fail_safe=True)
