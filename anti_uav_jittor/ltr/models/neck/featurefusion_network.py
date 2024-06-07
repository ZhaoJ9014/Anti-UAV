# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
TransT FeatureFusionNetwork class.

Copy-paste from torch.nn.Transformer with modifications:
    * positional encodings are passed in MHattention
    * extra LN at the end of encoder is removed
    * decoder returns a stack of activations from all decoding layers
"""
import copy
from typing import Optional

import jittor as jt
import jittor.nn as nn


class MultiheadAttention(jt.nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.0):
        super(MultiheadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        self.q_proj_weight = jt.nn.Parameter(jt.empty(embed_dim, embed_dim))
        self.k_proj_weight = jt.nn.Parameter(jt.empty(embed_dim, embed_dim))
        self.v_proj_weight = jt.nn.Parameter(jt.empty(embed_dim, embed_dim))

        self.out_proj = jt.nn.Linear(embed_dim, embed_dim)
        self._reset_parameters()

    def _reset_parameters(self):
        jt.nn.init.xavier_uniform_(self.q_proj_weight)
        jt.nn.init.xavier_uniform_(self.k_proj_weight)
        jt.nn.init.xavier_uniform_(self.v_proj_weight)
        jt.nn.init.xavier_uniform_(self.out_proj.weight)

    def execute(self, query, key, value, need_weights=True):
        # Step 1: Linear projections
        q = jt.nn.linear(query, self.q_proj_weight)
        k = jt.nn.linear(key, self.k_proj_weight)
        v = jt.nn.linear(value, self.v_proj_weight)

        # Step 2: Reshape for multi-head attention
        batch_size = query.size(1)
        q = q.contiguous().view(-1, batch_size * self.num_heads, self.head_dim).transpose(0, 1)
        k = k.contiguous().view(-1, batch_size * self.num_heads, self.head_dim).transpose(0, 1)
        v = v.contiguous().view(-1, batch_size * self.num_heads, self.head_dim).transpose(0, 1)

        # Step 3: Scaled dot-product attention
        attn_output, attn_output_weights = self.scaled_dot_product_attention(q, k, v)

        # Step 4: Concatenate heads and put through final linear layer
        attn_output = attn_output.transpose(0, 1).contiguous().view(-1, batch_size, self.embed_dim)
        attn_output = self.out_proj(attn_output)

        if need_weights:
            return attn_output, attn_output_weights
        else:
            return attn_output, None

    def scaled_dot_product_attention(self, q, k, v):
        d_k = q.size(-1)
        d_k = jt.Var(d_k)
        d_k=d_k.astype(jt.float32)
        #scores = jt.matmul(q, k.transpose(-2, -1)) / jt.sqrt(jt.Var(d_k, dtype=jt.float32))
        scores = jt.matmul(q, k.transpose(-2, -1)) / jt.sqrt(d_k)
        attn_weights = jt.nn.softmax(scores, dim=-1)
        attn_weights = jt.nn.dropout(attn_weights, p=self.dropout, is_train=self.training)
        output = jt.nn.matmul(attn_weights, v)
        return output, attn_weights


class FeatureFusionNetwork(jt.nn.Module):
    def __init__(self, d_model=256, nhead=8, num_featurefusion_layers=4,
                 dim_feedexecute=2048, dropout=0.1, activation="relu"):
        super().__init__()

        featurefusion_layer = FeatureFusionLayer(d_model, nhead, dim_feedexecute, dropout, activation)
        self.encoder = Encoder(featurefusion_layer, num_featurefusion_layers)

        decoderCFA_layer = DecoderCFALayer(d_model, nhead, dim_feedexecute, dropout, activation)
        decoderCFA_norm = jt.nn.LayerNorm(d_model)
        self.decoder = Decoder(decoderCFA_layer, decoderCFA_norm)

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                jt.nn.init.xavier_uniform_(p)

    def execute(self, src_temp, mask_temp, src_search, mask_search, pos_temp, pos_search):
        # src_temp = src_temp.flatten(2).permute(2, 0, 1)
        # pos_temp = pos_temp.flatten(2).permute(2, 0, 1)
        # src_search = src_search.flatten(2).permute(2, 0, 1)
        # pos_search = pos_search.flatten(2).permute(2, 0, 1)

        # mask_temp = mask_temp.flatten(1)
        # mask_search = mask_search.flatten(1)

        memory_temp, memory_search = self.encoder(src1=src_temp, src2=src_search,
                                                  src1_key_padding_mask=None,
                                                  src2_key_padding_mask=None,
                                                  pos_src1=pos_temp,
                                                  pos_src2=pos_search)
        hs = self.decoder(memory_search, memory_temp,
                          tgt_key_padding_mask=None,
                          memory_key_padding_mask=None,
                          pos_enc=pos_temp, pos_dec=pos_search)
        return hs.transpose(0, 1)


class Decoder(jt.nn.Module):

    def __init__(self, decoderCFA_layer, norm=None):
        super().__init__()
        self.layers = _get_clones(decoderCFA_layer, 1)
        self.norm = norm

    def execute(self, tgt, memory,
                tgt_mask: Optional[jt.Var] = None,
                memory_mask: Optional[jt.Var] = None,
                tgt_key_padding_mask: Optional[jt.Var] = None,
                memory_key_padding_mask: Optional[jt.Var] = None,
                pos_enc: Optional[jt.Var] = None,
                pos_dec: Optional[jt.Var] = None):
        output = tgt

        for layer in self.layers:
            output = layer(output, memory, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos_enc=pos_enc, pos_dec=pos_dec)

        if self.norm is not None:
            output = self.norm(output)

        return output

class Encoder(nn.Module):

    def __init__(self, featurefusion_layer, num_layers):
        super().__init__()
        self.layers = _get_clones(featurefusion_layer, num_layers)
        self.num_layers = num_layers

    def execute(self, src1, src2,
                src1_mask: Optional[jt.Var] = None,
                src2_mask: Optional[jt.Var] = None,
                src1_key_padding_mask: Optional[jt.Var] = None,
                src2_key_padding_mask: Optional[jt.Var] = None,
                pos_src1: Optional[jt.Var] = None,
                pos_src2: Optional[jt.Var] = None):
        output1 = src1
        output2 = src2

        for layer in self.layers:
            output1, output2 = layer(output1, output2, src1_mask=src1_mask,
                                     src2_mask=src2_mask,
                                     src1_key_padding_mask=src1_key_padding_mask,
                                     src2_key_padding_mask=src2_key_padding_mask,
                                     pos_src1=pos_src1, pos_src2=pos_src2)

        return output1, output2


class DecoderCFALayer(jt.nn.Module):

    def __init__(self, d_model, nhead, dim_feedexecute=2048, dropout=0.1, activation="relu"):
        super().__init__()

        self.multihead_attn =  MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedexecute model
        self.linear1 = jt.nn.Linear(d_model, dim_feedexecute)
        self.dropout = jt.nn.Dropout(dropout)
        self.linear2 = jt.nn.Linear(dim_feedexecute, d_model)

        self.norm1 = jt.nn.LayerNorm(d_model)
        self.norm2 = jt.nn.LayerNorm(d_model)

        self.dropout1 = jt.nn.Dropout(dropout)
        self.dropout2 = jt.nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def with_pos_embed(self, tensor, pos: Optional[jt.Var]):
        return tensor if pos is None else tensor + pos

    def execute_post(self, tgt, memory,
                     tgt_mask: Optional[jt.Var] = None,
                     memory_mask: Optional[jt.Var] = None,
                     tgt_key_padding_mask: Optional[jt.Var] = None,
                     memory_key_padding_mask: Optional[jt.Var] = None,
                     pos_enc: Optional[jt.Var] = None,
                     pos_dec: Optional[jt.Var] = None):

        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, pos_dec),
                                   key=self.with_pos_embed(memory, pos_enc),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        return tgt


    def execute(self, tgt, memory,
                tgt_mask: Optional[jt.Var] = None,
                memory_mask: Optional[jt.Var] = None,
                tgt_key_padding_mask: Optional[jt.Var] = None,
                memory_key_padding_mask: Optional[jt.Var] = None,
                pos_enc: Optional[jt.Var] = None,
                pos_dec: Optional[jt.Var] = None):

        return self.execute_post(tgt, memory, tgt_mask, memory_mask,
                                 tgt_key_padding_mask, memory_key_padding_mask, pos_enc, pos_dec)

class FeatureFusionLayer(jt.nn.Module):

    def __init__(self, d_model, nhead, dim_feedexecute=2048, dropout=0.1,
                 activation="relu"):
        super().__init__()
        self.self_attn1 = MultiheadAttention(d_model, nhead, dropout=dropout)
        self.self_attn2 = MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn1 = MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn2 = MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedexecute model
        self.linear11 = jt.nn.Linear(d_model, dim_feedexecute)
        self.dropout1 = jt.nn.Dropout(dropout)
        self.linear12 = jt.nn.Linear(dim_feedexecute, d_model)

        self.linear21 = jt.nn.Linear(d_model, dim_feedexecute)
        self.dropout2 = jt.nn.Dropout(dropout)
        self.linear22 = jt.nn.Linear(dim_feedexecute, d_model)

        self.norm11 = jt.nn.LayerNorm(d_model)
        self.norm12 = jt.nn.LayerNorm(d_model)
        self.norm13 = jt.nn.LayerNorm(d_model)
        self.norm21 = jt.nn.LayerNorm(d_model)
        self.norm22 = jt.nn.LayerNorm(d_model)
        self.norm23 = jt.nn.LayerNorm(d_model)
        self.dropout11 = jt.nn.Dropout(dropout)
        self.dropout12 =jt.nn.Dropout(dropout)
        self.dropout13 = jt.nn.Dropout(dropout)
        self.dropout21 = jt.nn.Dropout(dropout)
        self.dropout22 = jt.nn.Dropout(dropout)
        self.dropout23 = jt.nn.Dropout(dropout)

        self.activation1 = _get_activation_fn(activation)
        self.activation2 = _get_activation_fn(activation)

    def with_pos_embed(self, tensor, pos: Optional[jt.Var]):
        return tensor if pos is None else tensor + pos

    def execute_post(self, src1, src2,
                     src1_mask: Optional[jt.Var] = None,
                     src2_mask: Optional[jt.Var] = None,
                     src1_key_padding_mask: Optional[jt.Var] = None,
                     src2_key_padding_mask: Optional[jt.Var] = None,
                     pos_src1: Optional[jt.Var] = None,
                     pos_src2: Optional[jt.Var] = None):
        ##ECA层
        q1 = k1 = self.with_pos_embed(src1, pos_src1)
        src12 = self.self_attn1(q1, k1, value=src1, attn_mask=src1_mask,
                               key_padding_mask=src1_key_padding_mask)[0]
        src1 = src1 + self.dropout11(src12)
        src1 = self.norm11(src1)

        q2 = k2 = self.with_pos_embed(src2, pos_src2)
        src22 = self.self_attn2(q2, k2, value=src2, attn_mask=src2_mask,
                               key_padding_mask=src2_key_padding_mask)[0]
        src2 = src2 + self.dropout21(src22)
        src2 = self.norm21(src2)


        src12 = self.multihead_attn1(query=self.with_pos_embed(src1, pos_src1),
                                   key=self.with_pos_embed(src2, pos_src2),
                                   value=src2, attn_mask=src2_mask,
                                   key_padding_mask=src2_key_padding_mask)[0]
        src22 = self.multihead_attn2(query=self.with_pos_embed(src2, pos_src2),
                                   key=self.with_pos_embed(src1, pos_src1),
                                   value=src1, attn_mask=src1_mask,
                                   key_padding_mask=src1_key_padding_mask)[0]

        src1 = src1 + self.dropout12(src12)
        src1 = self.norm12(src1)
        src12 = self.linear12(self.dropout1(self.activation1(self.linear11(src1))))
        src1 = src1 + self.dropout13(src12)
        src1 = self.norm13(src1)

        src2 = src2 + self.dropout22(src22)
        src2 = self.norm22(src2)
        src22 = self.linear22(self.dropout2(self.activation2(self.linear21(src2))))
        src2 = src2 + self.dropout23(src22)
        src2 = self.norm23(src2)

        return src1, src2

    def execute(self, src1, src2,
                src1_mask: Optional[jt.Var] = None,
                src2_mask: Optional[jt.Var] = None,
                src1_key_padding_mask: Optional[jt.Var] = None,
                src2_key_padding_mask: Optional[jt.Var] = None,
                pos_src1: Optional[jt.Var] = None,
                pos_src2: Optional[jt.Var] = None):

        return self.execute_post(src1, src2, src1_mask, src2_mask,
                                 src1_key_padding_mask, src2_key_padding_mask, pos_src1, pos_src2)


def _get_clones(module, N):
    return jt.nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def build_featurefusion_network(settings):
     model = FeatureFusionNetwork(
        d_model=512,
        dropout=settings.dropout,
        nhead=settings.nheads,
        dim_feedexecute=settings.dim_feedexecute,
        num_featurefusion_layers=settings.featurefusion_layers
    )
     return model

def glu(input):
    # 假设输入的最后一个维度是偶数，可以平分成两半
    split_size = input.size(-1) // 2
    # 将输入沿着最后一个维度分成两半
    a, b = input.split(split_size, dim=-1)
    # 使用第一半作为门控信号
    return a * jt.sigmoid(b)
def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return jt.nn.relu
    if activation == "gelu":
        return jt.nn.gelu
    if activation == "glu":
        return glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")

