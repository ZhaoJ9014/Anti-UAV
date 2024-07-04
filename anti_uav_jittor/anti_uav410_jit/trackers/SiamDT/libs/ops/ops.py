# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.distributed as dist
import jittor as jt
import jittor.nn as nn

import os
from multiprocessing.pool import ThreadPool as Pool


def fast_xcorr(z, x):
    r'''Fast cross-correlation.
    '''
    assert z.shape[1] == x.shape[1]
    nz = z.size(0)
    nx, c, h, w = x.size()
    x = x.view(-1, nz * c, h, w)
    out = nn.conv2d(x, z, groups=nz)
    out = out.view(nx, -1, out.size(-2), out.size(-1))
    return out


def init_weights(model, gain=1):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight, gain)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight, gain)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


def kaiming_init(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif isinstance(m, (nn.Conv1d, nn.Conv2d)):
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)) and m.affine:
        nn.init.constant_(m.weight, 1.0)
        nn.init.constant_(m.bias, 0.0)


def classifier_init(m):
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, std=0.001)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)


# def get_dist_info():
#     if torch.__version__ < '1.0':
#         initialized = dist._initialized
#     else:
#         initialized = dist.is_initialized()
#     if initialized:
#         rank = dist.get_rank()
#         world_size = dist.get_world_size()
#     else:
#         rank = 0
#         world_size = 1
#     return rank, world_size


def put_device(data, device, non_blocking=False):
    non_blocking = non_blocking if 'cuda' in device.type else False
    if isinstance(data, jt.Var):
        data = data.to(device, non_blocking=non_blocking)
    elif isinstance(data, (list, tuple)):
        data = data.__class__([
            put_device(item, device, non_blocking=non_blocking)
            for item in data])
    elif isinstance(data, dict):
        data = {k: put_device(v, device, non_blocking=non_blocking)
                for k, v in data.items()}
    return data


def adaptive_apply(func, args):
    if isinstance(args, (tuple, list)):
        return func(*args)
    elif isinstance(args, dict):
        return func(**args)
    else:
        return func(args)


def detach(data):
    if isinstance(data, jt.Var):
        return data.detach()
    elif isinstance(data, (list, tuple)):
        data = data.__class__([detach(item) for item in data])
    elif isinstance(data, dict):
        data = {k: detach(v) for k, v in data.items()}
    return data


def map(func, args_list, num_workers=32, timeout=1):
    if isinstance(args_list, range):
        args_list = list(args_list)
    assert isinstance(args_list, list)
    if not isinstance(args_list[0], tuple):
        args_list = [(args, ) for args in args_list]
    
    with Pool(processes=num_workers) as pool:
        results = [pool.apply_async(func, args) for args in args_list]
        results = [res.get(timeout=timeout) for res in results]
    
    return results
