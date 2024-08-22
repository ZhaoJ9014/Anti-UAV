import argparse
import jittor as jt
import os
import time
import importlib
import _init_paths
from jittor import nn
from lib.models.mixformer.mixformer_online import Attention
from thop import profile
from thop.utils import clever_format

def parse_args():
    """
    args for training.
    """
    parser = argparse.ArgumentParser(description='Parse args for training')
    # for train
    parser.add_argument('--script', type=str, default='mixformer_online', choices=['mixformer', 'mixformer_online'],
                        help='training script name')
    parser.add_argument('--config', type=str, default='baseline', help='yaml configure file name')
    parser.add_argument('--display_name', type=str, default='MixFormer')
    parser.add_argument('--online_skip', type=int, default=200, help='the skip interval of mixformer-online')
    args = parser.parse_args()

    return args

def get_complexity_MHA(m:nn.MultiheadAttention, x, y):
    """(L, B, D): sequence length, batch size, dimension"""
    d_mid = m.embed_dim
    query, key, value = x[0], x[1], x[2]
    Lq, batch, d_inp = query.size()
    Lk = key.size(0)
    """compute flops"""
    total_ops = 0
    # projection of Q, K, V
    total_ops += d_inp * d_mid * Lq * batch  # query
    total_ops += d_inp * d_mid * Lk * batch * 2  # key and value
    # compute attention
    total_ops += Lq * Lk * d_mid * 2
    m.total_ops += jt.Var([int(total_ops)])

def get_complexity_Attention(module: Attention, input, y):
    # T: num_token
    # S: num_token
    input, H_t, W_t, H_s, W_s = input
    # input_t, input_s =

    flops = 0

    _, T, C = input.shape
    # H = W = int(np.sqrt(T))

    H_Q_t = H_t / module.stride_q
    W_Q_t = H_t / module.stride_q
    T_Q_t = H_Q_t * W_Q_t

    H_KV_t = H_t / module.stride_kv
    W_KV_t = W_t / module.stride_kv
    T_KV_t = H_KV_t * W_KV_t

    H_Q_s = H_s / module.stride_q
    W_Q_s = H_s / module.stride_q
    T_Q_s = H_Q_s * W_Q_s

    H_KV_s = H_s / module.stride_kv
    W_KV_s = W_s / module.stride_kv
    T_KV_s = H_KV_s * W_KV_s

    T_Q = T_Q_t * 2 + T_Q_s  # including template， online template and search
    T_KV = T_KV_t * 2 + T_KV_s

    # # Scaled-dot-product macs
    # # [B x T x C] x [B x C x T] --> [B x T x S]
    # # multiplication-addition is counted as 1 because operations can be fused
    # flops += T_Q * T_KV * module.dim
    # # [B x T x S] x [B x S x C] --> [B x T x C]
    # flops += T_Q * module.dim * T_KV

    # when use asymmetric mixed attention
    flops += T_Q_s * T_KV * module.dim
    flops += T_Q_s * module.dim * T_KV
    flops += T_Q_t * T_KV_t * module.dim * 2  # including template and online template
    flops += T_Q_t * module.dim * T_KV_t * 2

    if (
            hasattr(module, 'conv_proj_q')
            and hasattr(module.conv_proj_q, 'conv')
    ):
        params = sum(
            [
                p.numel()
                for p in module.conv_proj_q.conv.parameters()
            ]
        )
        flops += params * H_Q_t * W_Q_t * 2
        flops += params * H_Q_s * W_Q_s

    if (
            hasattr(module, 'conv_proj_k')
            and hasattr(module.conv_proj_k, 'conv')
    ):
        params = sum(
            [
                p.numel()
                for p in module.conv_proj_k.conv.parameters()
            ]
        )
        flops += params * H_KV_t * W_KV_t * 2
        flops += params * H_KV_s * W_KV_s

    if (
            hasattr(module, 'conv_proj_v')
            and hasattr(module.conv_proj_v, 'conv')
    ):
        params = sum(
            [
                p.numel()
                for p in module.conv_proj_v.conv.parameters()
            ]
        )
        flops += params * H_KV_t * W_KV_t * 2
        flops += params * H_KV_s * W_KV_s

    params = sum([p.numel() for p in module.proj_q.parameters()])
    flops += params * T_Q
    params = sum([p.numel() for p in module.proj_k.parameters()])
    flops += params * T_KV
    params = sum([p.numel() for p in module.proj_v.parameters()])
    flops += params * T_KV
    params = sum([p.numel() for p in module.proj.parameters()])
    flops += params * T

    module.total_ops += jt.Var([int(flops)])


def evaluate(model, template, search, skip=200, display_info='MixFormer'):
    """Compute FLOPs, Params, and Speed"""
    # compute flops and params except for score prediction
    custom_ops = {Attention: get_complexity_Attention}
    macs, params = profile(model, inputs=(template, template, search, False),
                           custom_ops=custom_ops, verbose=False)
    macs, params = clever_format([macs, params], "%.3f")
    print('==>Macs is ', macs)
    print('==>Params is ', params)

    # test speed
    T_w = 10
    T_t = 1000
    print("testing speed ...")
    with jt.no_grad():
        # overall
        for i in range(T_w):
            _ = model(template, template, search, run_score_head=True)
        start = time.time()
        for i in range(T_t):
            if i % skip == 0:
                _ = model.set_online(template, template)
            _ = model.forward_test(search)
        end = time.time()
        avg_lat = (end - start) / T_t
        print("\033[0;32;40m The average overall FPS of {} is {}.\033[0m" .format(display_info, 1.0/avg_lat))


def get_data(bs, sz):
    img_patch = jt.randn(bs, 3, sz, sz)
    return img_patch


if __name__ == "__main__":
    device = "cuda:0"
    # torch.cuda.set_device(device)
    args = parse_args()
    '''update cfg'''
    prj_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    yaml_fname = os.path.join(prj_dir, 'experiments/%s/%s.yaml' % (args.script, args.config))
    print("yaml_fname: {}".format(yaml_fname))
    config_module = importlib.import_module('lib.config.%s.config' % args.script)
    cfg = config_module.cfg
    config_module.update_config_from_file(yaml_fname)
    print("cfg: {}".format(cfg))
    '''set some values'''
    bs = 1
    z_sz = cfg.TEST.TEMPLATE_SIZE
    x_sz = cfg.TEST.SEARCH_SIZE
    cfg.MODEL.BACKBONE.FREEZE_BN = False
    cfg.MODEL.HEAD_FREEZE_BN = False
    '''import stark network module'''
    model_module = importlib.import_module('lib.models.mixformer')
    if args.script == "mixformer_online":
        model_constructor = model_module.build_mixformer_online_score
        model = model_constructor(cfg)
        # get the template and search
        template = get_data(bs, z_sz)
        search = get_data(bs, x_sz)
        # transfer to device
        model = model.to(device)
        template = template.to(device)
        search = search.to(device)
        # evaluate the model properties
        evaluate(model, template, search, args.online_skip, args.display_name)
