# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------
# https://github.com/microsoft/Swin-Transformer/blob/main/models/swin_transformer.py
# https://github.com/SwinTransformer/Swin-Transformer-Object-Detection/blob/master/mmdet/models/backbones/swin_transformer.py

import math

from jittor import nn
import jittor as jt
from itertools import repeat
import collections.abc

def _trunc_normal_(tensor, mean, std, a, b):
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.
    l = norm_cdf((a - mean) / std)
    u = norm_cdf((b - mean) / std)

    tensor.uniform_(2 * l - 1, 2 * u - 1)


    tensor.erfinv_()

    # Transform to proper mean, std
    tensor.mul_(std * math.sqrt(2.))
    tensor.add_(mean)

    # Clamp to ensure it's in the proper range
    tensor.clamp_(min=a, max=b)
    return tensor

def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    return _trunc_normal_(tensor, mean, std, a, b)
def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
            return tuple(x)
        return tuple(repeat(x, n))
    return parse


to_2tuple = _ntuple(2)


def drop_path(x, drop_prob: float = 0., training: bool = False, scale_by_keep: bool = True):

    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob: float = 0., scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)

    def extra_repr(self):
        return f'drop_prob={round(self.drop_prob,3):0.3f}'

class Mlp(nn.Module):
    """ Multilayer perceptron."""

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size
    Returns:
        test_d: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image
    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


def _generate_2d_relative_position_index(size):
    # get siam_pair-wise relative position index for each token inside the window
    coords_h = jt.arange(size[0])
    coords_w = jt.arange(size[1])
    coords = jt.stack(jt.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
    coords_flatten = jt.flatten(coords, 1)  # 2, Wh*Ww
    relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
    relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
    relative_coords[:, :, 0] += size[0] - 1  # shift to start from 0
    relative_coords[:, :, 1] += size[1] - 1
    relative_coords[:, :, 0] *= 2 * size[1] - 1
    relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
    return relative_position_index


class WindowAttention(nn.Module):
    """ Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.
    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            jt.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        relative_position_index = _generate_2d_relative_position_index(self.window_size)
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """ Forward function.
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SwinTransformerBlock(nn.Module):
    """ Swin Transformer Block.
    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.H = None
        self.W = None

    def forward(self, x, mask_matrix=None):
        """ Forward function.
        Args:
            x: Input feature, tensor size (B, H*W, C).
            H, W: Spatial resolution of the input feature.
            mask_matrix: Attention mask for cyclic shift.
        """
        B, L, C = x.shape
        H, W = self.H, self.W
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # pad feature maps to multiples of window size
        pad_l = pad_t = 0
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        x = nn.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
        _, Hp, Wp, _ = x.shape

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = jt.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            attn_mask = mask_matrix
        else:
            shifted_x = x
            attn_mask = None

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=attn_mask)  # nW*B, window_size*window_size, C

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, Hp, Wp)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = jt.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x

        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :].contiguous()

        x = x.view(B, H * W, C)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    Args:
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None, pretrain_img_size=224,
                 ape=False, drop_rate=0.):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        self.patch_size = patch_size

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

        self.ape = ape

        # absolute position embedding
        if self.ape:
            pretrain_img_size = to_2tuple(pretrain_img_size)
            patch_size = to_2tuple(patch_size)
            patches_resolution = [pretrain_img_size[0] // patch_size[0], pretrain_img_size[1] // patch_size[1]]

            self.absolute_pos_embed = nn.Parameter(
                jt.zeros(1, embed_dim, patches_resolution[0], patches_resolution[1]))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

    def forward(self, x, H, W):
        """Forward function."""
        # padding
        if W % self.patch_size[1] != 0:
            x = nn.pad(x, (0, self.patch_size[1] - W % self.patch_size[1]))
        if H % self.patch_size[0] != 0:
            x = nn.pad(x, (0, 0, 0, self.patch_size[0] - H % self.patch_size[0]))

        x = self.proj(x)  # B C Wh Ww

        Wh, Ww = x.size(2), x.size(3)

        if self.norm is not None:
            x = x.flatten(2).transpose(1, 2)
            x = self.norm(x)
            x = x.transpose(1, 2).view(-1, self.embed_dim, Wh, Ww)

        if self.ape:
            # interpolate the position embedding to the corresponding size
            absolute_pos_embed = F.interpolate(self.absolute_pos_embed, size=(Wh, Ww), mode='bicubic')
            x = (x + absolute_pos_embed).flatten(2).transpose(1, 2)  # B Wh*Ww C
        else:
            x = x.flatten(2).transpose(1, 2)
        x = self.pos_drop(x)

        return x, Wh, Ww


class PatchMerging(nn.Module):
    """ Patch Merging Layer
    Args:
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x, H, W):
        """ Forward function.
        Args:
            x: Input feature, tensor size (B, H*W, C).
            H, W: Spatial resolution of the input feature.
        """
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        x = x.view(B, H, W, C)

        # padding
        pad_input = (H % 2 == 1) or (W % 2 == 1)
        if pad_input:
            x = nn.pad(x, (0, 0, 0, W % 2, 0, H % 2))

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = jt.concat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

        Wh, Ww = (H + 1) // 2, (W + 1) // 2

        return x, Wh, Ww


class BasicStage(nn.Module):
    """ A basic Swin Transformer layer for one stage.
    Args:
        dim (int): Number of feature channels
        depth (int): Depths of this stage.
        num_heads (int): Number of attention head.
        window_size (int): Local window size. Default: 7.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self,
                 dim,
                 depth,
                 num_heads,
                 window_size=7,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 norm_layer=nn.LayerNorm,
                 pre_stage=None,
                 use_checkpoint=False):
        super().__init__()
        self.window_size = window_size
        self.shift_size = window_size // 2
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim=dim,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer)
            for i in range(depth)])

        if pre_stage is not None:
            self.pre_stage = pre_stage

    def forward(self, x, H, W):
        """ Forward function.
        Args:
            x: Input feature, tensor size (B, H*W, C).
            H, W: Spatial resolution of the input feature.
        """

        if self.pre_stage is not None:
            x, H, W = self.pre_stage(x, H, W)

        # calculate attention mask for SW-MSA
        Hp = int(math.ceil(H / self.window_size)) * self.window_size
        Wp = int(math.ceil(W / self.window_size)) * self.window_size
        img_mask = jt.zeros((1, Hp, Wp, 1), device=x.device)  # 1 Hp Wp 1
        h_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

        for blk in self.blocks:
            blk.H, blk.W = H, W
            # if self.use_checkpoint:
            #     x = checkpoint.checkpoint(blk, x, attn_mask)
            # else:
            x = blk(x, attn_mask)

        return x, H, W


def _build_stages(num_layers,
                  pretrain_img_size=224,
                  patch_size=4,
                  in_chans=3,
                  embed_dim=96,
                  depths=(2, 2, 6, 2),
                  num_heads=(3, 6, 12, 24),
                  window_size=7,
                  mlp_ratio=4.,
                  qkv_bias=True,
                  qk_scale=None,
                  drop_rate=0.,
                  attn_drop_rate=0.,
                  drop_path_rate=0.2,
                  norm_layer=nn.LayerNorm,
                  ape=False,
                  patch_norm=True,
                  use_checkpoint=False):
    # build layers
    stages = nn.ModuleList()
    dpr = [x.item() for x in jt.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
    num_features = []
    for i_layer in range(num_layers):
        if i_layer == 0:
            pre_stage = PatchEmbed(
                patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
                norm_layer=norm_layer if patch_norm else None, pretrain_img_size=pretrain_img_size, ape=ape,
                drop_rate=drop_rate)
        else:
            dim = int(embed_dim * 2 ** (i_layer - 1))
            pre_stage = PatchMerging(dim, norm_layer)

        dim = int(embed_dim * 2 ** i_layer)
        stage = BasicStage(
            dim=int(embed_dim * 2 ** i_layer),
            depth=depths[i_layer],
            num_heads=num_heads[i_layer],
            window_size=window_size,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop=drop_rate,
            attn_drop=attn_drop_rate,
            drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
            norm_layer=norm_layer,
            pre_stage=pre_stage,
            use_checkpoint=use_checkpoint)
        stages.append(stage)
        num_features.append(dim)
    return stages, num_features


def _freeze_stages(stages, frozen_stages, ape):
    patch_embed = stages[0].pre_stage
    if frozen_stages >= 0:
        patch_embed.proj.eval()
        for param in patch_embed.proj.parameters():
            param.requires_grad = False
        if patch_embed.norm is not None:
            patch_embed.norm.eval()
            for param in patch_embed.norm.parameters():
                param.requires_grad = False

    if frozen_stages >= 1 and ape:
        patch_embed.absolute_pos_embed.requires_grad = False

    if frozen_stages >= 2:
        patch_embed.pos_drop.eval()
        for i in range(0, frozen_stages - 1):
            m = stages[i].blocks
            m.eval()
            for param in m.parameters():
                param.requires_grad = False
            if i + 1 != len(stages):
                m = stages[i + 1].pre_stage
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False


def _update_state_dict_(state_dict, prefix=''):
    for full_key in list(state_dict.keys()):
        key: str = full_key[len(prefix):]
        if key.startswith('layers.'):
            layer_index = int(key[len('layers.'): len('layers.') + 1])
            if key[len('layers..') + 1:].startswith('downsample'):
                key_rest = key[len('layers..downsample') + 1:]
                state_dict[f'{prefix}stages.{layer_index + 1}.pre_stage{key_rest}'] = state_dict.pop(full_key)
            else:
                state_dict[full_key.replace('layers', 'stages')] = state_dict.pop(full_key)
        elif key.startswith('patch_embed'):
            state_dict[full_key.replace('patch_embed', 'stages.0.pre_stage')] = state_dict.pop(full_key)


class SwinTransformer(nn.Module):
    def __init__(self,
                 pretrain_img_size=224,
                 patch_size=4,
                 in_chans=3,
                 embed_dim=96,
                 depths=(2, 2, 6, 2),
                 num_heads=(3, 6, 12, 24),
                 window_size=7,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.2,
                 norm_layer=nn.LayerNorm,
                 ape=False,
                 patch_norm=True,
                 out_indices=(0, 1, 2, 3),
                 frozen_stages=-1,
                 use_checkpoint=False):
        super(SwinTransformer, self).__init__()
        self.stages, self.stage_dims = _build_stages(len(depths), pretrain_img_size,
                                                     patch_size,
                                                     in_chans,
                                                     embed_dim,
                                                     depths,
                                                     num_heads,
                                                     window_size,
                                                     mlp_ratio,
                                                     qkv_bias,
                                                     qk_scale,
                                                     drop_rate,
                                                     attn_drop_rate,
                                                     drop_path_rate,
                                                     norm_layer,
                                                     ape,
                                                     patch_norm,
                                                     use_checkpoint)

        self.num_channels_output = tuple(self.stage_dims[i] for i in out_indices)
        # add a norm layer for each output
        for i_layer in out_indices:
            layer = norm_layer(self.stage_dims[i_layer])
            layer_name = f'norm{i_layer}'
            self.add_module(layer_name, layer)
        self.num_stages = max(out_indices) + 1
        self.out_indices = out_indices

        self.frozen_stages = frozen_stages
        self.ape = ape
        _freeze_stages(self.stages, self.frozen_stages, self.ape)

    def init_weights(self, pretrained=None):
        def _init_weights(m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

        # if isinstance(pretrained, str):
        #     self.apply(_init_weights)
        #     state_dict = torch.hub.load_state_dict_from_url(pretrained, map_location='cpu')['model']
        #     _update_state_dict_(state_dict)
        #     self.load_state_dict(state_dict, strict=False)
        # elif pretrained is None:
        #     self.apply(_init_weights)
        # else:
        #     raise TypeError('pretrained must be a str or None')

    def forward(self, x, out_indices=None, reshape=True):
        if out_indices is None:
            out_indices = self.out_indices
        _, _, H, W = x.size()
        outs = []
        for i in range(self.num_stages):
            layer = self.stages[i]
            x, H, W = layer(x, H, W)
            if i in out_indices:
                norm_layer = getattr(self, f'norm{i}')
                x_out = norm_layer(x)
                if reshape:
                    x_out = x_out.view(-1, H, W, self.stage_dims[i]).permute(0, 3, 1, 2).contiguous()
                outs.append(x_out)
        return outs

    def train(self, mode=True):
        """Convert the model into training mode while keep layers freezed."""
        super(SwinTransformer, self).train(mode)
        _freeze_stages(self.stages, self.frozen_stages, self.ape)


_cfg = {
    'swin_base_patch4_window12_384': dict(
        url='https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window12_384_22kto1k.pth',
        params=dict(pretrain_img_size=384, patch_size=4, window_size=12, embed_dim=128, depths=(2, 2, 18, 2),
                    num_heads=(4, 8, 16, 32))
    ),
    'swin_base_patch4_window7_224': dict(
        url='https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window7_224_22kto1k.pth',
        params=dict(pretrain_img_size=224, patch_size=4, window_size=7, embed_dim=128, depths=(2, 2, 18, 2),
                    num_heads=(4, 8, 16, 32))
    ),

    'swin_large_patch4_window12_384': dict(
        url='https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_large_patch4_window12_384_22kto1k.pth',
        params=dict(pretrain_img_size=384, patch_size=4, window_size=12, embed_dim=192, depths=(2, 2, 18, 2),
                    num_heads=(6, 12, 24, 48))
    ),

    'swin_large_patch4_window7_224': dict(
        url='https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_large_patch4_window7_224_22kto1k.pth',
        params=dict(pretrain_img_size=224, patch_size=4, window_size=7, embed_dim=192, depths=(2, 2, 18, 2),
                    num_heads=(6, 12, 24, 48))
    ),

    'swin_small_patch4_window7_224': dict(
        url='https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_small_patch4_window7_224.pth',
        params=dict(pretrain_img_size=224, patch_size=4, window_size=7, embed_dim=96, depths=(2, 2, 18, 2),
                    num_heads=(3, 6, 12, 24))
    ),

    'swin_tiny_patch4_window7_224': dict(
        url='https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth',
        params=dict(pretrain_img_size=224, patch_size=4, window_size=7, embed_dim=96, depths=(2, 2, 6, 2),
                    num_heads=(3, 6, 12, 24))
    ),

    'swin_base_patch4_window12_384_in22k': dict(
        url='https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window12_384_22k.pth',
        params=dict(pretrain_img_size=384, patch_size=4, window_size=12, embed_dim=128, depths=(2, 2, 18, 2),
                    num_heads=(4, 8, 16, 32))
    ),

    'swin_base_patch4_window7_224_in22k': dict(
        url='https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window7_224_22k.pth',
        params=dict(pretrain_img_size=224, patch_size=4, window_size=7, embed_dim=128, depths=(2, 2, 18, 2),
                    num_heads=(4, 8, 16, 32))
    ),

    'swin_large_patch4_window12_384_in22k': dict(
        url='https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_large_patch4_window12_384_22k.pth',
        params=dict(pretrain_img_size=224, patch_size=4, window_size=12, embed_dim=192, depths=(2, 2, 18, 2),
                    num_heads=(6, 12, 24, 48))
    ),

    'swin_large_patch4_window7_224_in22k': dict(
        url='https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_large_patch4_window7_224_22k.pth',
        params=dict(pretrain_img_size=224, patch_size=4, window_size=7, embed_dim=192, depths=(2, 2, 18, 2),
                    num_heads=(6, 12, 24, 48))
    )
}


def build_swin_transformer_backbone(name, load_pretrained=True, output_layers=(3,), frozen_stages=-1,
                                    overwrite_embed_dim=None):
    import copy

    pretrained_model_path = None
    if load_pretrained and 'url' in _cfg[name]:
        pretrained_model_path = _cfg[name]['url']

    max_output_index = max(output_layers)
    params = copy.deepcopy(_cfg[name]['params'])
    if max_output_index < 3:
        params['depths'] = params['depths'][0: max_output_index + 1]
        params['num_heads'] = params['num_heads'][0: max_output_index + 1]

    if overwrite_embed_dim is not None:
        params['embed_dim'] = overwrite_embed_dim
        pretrained_model_path = None
    transformer = SwinTransformer(out_indices=output_layers, frozen_stages=frozen_stages, **params)

    transformer.init_weights(pretrained_model_path)
    return transformer





'''
torch.Size([8, 192, 64, 64])
torch.Size([8, 384, 32, 32])
torch.Size([8, 768, 16, 16])
'''

# x = torch.rand(8,3,512,512)
# params = _cfg['swin_tiny_patch4_window7_224']['params']
# backbone = SwinTransformer(out_indices=(1,2,3), frozen_stages=-1, **params)
# y = backbone(x)
# for i in range(len(y)):
#     print(y[i].shape)
