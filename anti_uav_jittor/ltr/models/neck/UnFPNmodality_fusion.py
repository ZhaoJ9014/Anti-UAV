import copy
from typing import Optional


from util.misc import  generate_mask
import jittor as jt
from jittor import nn
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

    def forward(self, query, key, value, need_weights=True):
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
        scores = jt.matmul(q, k.transpose(-2, -1)) / jt.sqrt(jt.var(d_k, dtype=jt.float32))
        attn_weights = jt.nn.softmax(scores, dim=-1)
        attn_weights = jt.nn.dropout(attn_weights, p=self.dropout, training=self.training)
        output = jt.nn.matmul(attn_weights, v)
        return output, attn_weights

class ModalFusionNetwork(jt.nn.Module):
    def __init__(self,d_model=256, nhead=8, dim_feedforward=2048, dropout=0.1):
        super(ModalFusionNetwork, self).__init__()
        self.sef_attn_ir = MultiheadAttention(d_model, num_heads=nhead, dropout=dropout)
        self.sef_attn_grb = MultiheadAttention(d_model, num_heads=nhead, dropout=dropout)
        self.sef_attn_fusion = MultiheadAttention(d_model*2,num_heads=nhead, dropout=dropout)
        self.multihead_attn1 = MultiheadAttention(d_model, num_heads=nhead, dropout=dropout)
        self.multihead_attn2 = MultiheadAttention(d_model, num_heads=nhead, dropout=dropout)
        self.droputout_1 = jt.nn.Dropout(dropout)
        self.droputout_2 = jt.nn.Dropout(dropout)
        self.normal1 = jt.nn.LayerNorm(d_model)
        self.normal2 = jt.nn.LayerNorm(d_model)
        self.linear1 = jt.nn.Linear(d_model*2,dim_feedforward)
        self.droputout_3 = jt.nn.Dropout(dropout)
        self.droputout_4 = jt.nn.Dropout(dropout)
        self.linear2 = jt.nn.Linear(dim_feedforward,d_model*2)

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                jt.nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[jt.var]):
        return tensor if pos is None else tensor + pos

    def forward(self, rgb, ir, pos_rgb, pos_ir,flatten=False):
        '''

        Args:
            rgb: torch.Size([batch_size, hidden, H, W])
            ir: torch.Size([batch_size, hidden, H, W])
            pos_rgb:torch.Size([batch_size, hidden, H, W])
            pos_ir:torch.Size([batch_size, hidden, H, W])

        Returns:
            result of modal feature with torch.Size([H*W*2, batch_size, hidden])
        '''
        if flatten:
            rgb = rgb.flatten(2).permute(2, 0, 1)
            ir = ir.flatten(2).permute(2, 0, 1)
            pos_rgb = pos_rgb.flatten(2).permute(2, 0, 1)
            pos_ir = pos_ir.flatten(2).permute(2, 0, 1)
        q1 = k1 = self.with_pos_embed(rgb, pos_rgb)
        src11 = self.sef_attn_grb(q1, k1, value=rgb)[0]
        q2 = k2 = self.with_pos_embed(ir, pos_ir)
        src22 = self.sef_attn_ir(q2, k2, value=ir)[0]
        src11 = src11 + self.droputout_1(src11)
        src11 = self.normal1(src11)
        src22 = src22 +self.droputout_2(src22)
        src22 = self.normal2(src22)

        cross1 = self.multihead_attn1(src11, src22, src22)[0]
        cross2 = self.multihead_attn2(src22, src11, src11)[0]

        fusion = torch.concat((cross1,cross2),dim=2)

        fusion_cross = self.sef_attn_fusion(fusion,fusion,fusion)[0]

        fusion_cross = self.linear1(fusion_cross)
        fusion_cross = jt.nn.relu(fusion_cross)
        fusion_cross = self.droputout_3(fusion_cross)
        fusion_cross = self.linear2(fusion_cross)
        fusion_cross = jt.nn.relu(fusion_cross)
        fusion_cross = self.droputout_4(fusion_cross)

        return fusion_cross



class ModalFusionMultiScale(nn.Module):
    def __init__(self,position_encodeing,scale_layer=3,d_model=256, nhead=8, dim_feedforward=2048, dropout=0.1):
        super(ModalFusionMultiScale, self).__init__()
        self.scale_layers = scale_layer
        self.fusionblock = jt.nn.ModuleList([ModalFusionNetwork() for i in range(scale_layer)])
        self.positon_encodeing = position_encodeing

    def forward(self,rgb,ir):
        fusion_out = []
        for i in range(self.scale_layers):
            # mask_rgb = generate_mask(rgb[i])
            # mask_ir = generate_mask(ir[i])
            position_encodings_ir = self.positon_encodeing(ir[i])
            position_encodings_rgb = self.positon_encodeing(rgb[i])
            fusion_outi = self.fusionblock[i](rgb[i],ir[i],position_encodings_rgb,position_encodings_ir)
            fusion_out.append(fusion_outi)
        return fusion_out


import copy
from typing import Optional



class FeatureFusionNetwork(jt.nn.Module):

    def __init__(self, d_model=512, nhead=8, num_featurefusion_layers=4,
                 dim_feedforward=2048, dropout=0.1, activation="relu"):
        super().__init__()

        featurefusion_layer = FeatureFusionLayer(d_model, nhead, dim_feedforward, dropout, activation)
        self.encoder = Encoder(featurefusion_layer, num_featurefusion_layers)

        decoderCFA_layer = DecoderCFALayer(d_model, nhead, dim_feedforward, dropout, activation)
        decoderCFA_norm = nn.LayerNorm(d_model)
        self.decoder = Decoder(decoderCFA_layer, decoderCFA_norm)

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                jt.nn.init.xavier_uniform_(p)

    def forward(self, src_temp, mask_temp, src_search, mask_search,
                pos_temp: Optional[jt.var],
                pos_search: Optional[jt.var]):


        mask_temp = mask_temp.flatten(1)
        mask_search = mask_search.flatten(1)

        memory_temp, memory_search = self.encoder(src1=src_temp, src2=src_search,
                                                  src1_key_padding_mask=mask_temp,
                                                  src2_key_padding_mask=mask_search,
                                                  pos_src1=pos_temp,
                                                  pos_src2=pos_search)

        hs = self.decoder(memory_search, memory_temp,
                          tgt_key_padding_mask=mask_search,
                          memory_key_padding_mask=mask_temp,
                          pos_enc=pos_temp, pos_dec=pos_search)
        return hs.transpose(0, 1)

class Decoder(nn.Module):

    def __init__(self, decoderCFA_layer, norm=None):
        super().__init__()
        self.layers = _get_clones(decoderCFA_layer, 1)
        self.norm = norm

    def forward(self, tgt, memory,
                tgt_mask: Optional[jt.var] = None,
                memory_mask: Optional[jt.var] = None,
                tgt_key_padding_mask: Optional[jt.var] = None,
                memory_key_padding_mask: Optional[jt.var] = None,
                pos_enc: Optional[jt.var] = None,
                pos_dec: Optional[jt.var] = None):
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

    def forward(self, src1, src2,
                src1_mask: Optional[jt.var] = None,
                src2_mask: Optional[jt.var] = None,
                src1_key_padding_mask: Optional[jt.var] = None,
                src2_key_padding_mask: Optional[jt.var] = None,
                pos_src1: Optional[jt.var] = None,
                pos_src2: Optional[jt.var] = None):
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

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super().__init__()

        self.multihead_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = jt.nn.Linear(d_model, dim_feedforward)
        self.dropout = jt.nn.Dropout(dropout)
        self.linear2 = jt.nn.Linear(dim_feedforward, d_model)

        self.norm1 = jt.nn.LayerNorm(d_model)
        self.norm2 = jt.nn.LayerNorm(d_model)

        self.dropout1 = jt.nn.Dropout(dropout)
        self.dropout2 = jt.nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def with_pos_embed(self, tensor, pos: Optional[jt.var]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     tgt_mask: Optional[jt.var] = None,
                     memory_mask: Optional[jt.var] = None,
                     tgt_key_padding_mask: Optional[jt.var] = None,
                     memory_key_padding_mask: Optional[jt.var] = None,
                     pos_enc: Optional[jt.var] = None,
                     pos_dec: Optional[jt.var] = None):

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


    def forward(self, tgt, memory,
                tgt_mask: Optional[jt.var] = None,
                memory_mask: Optional[jt.var] = None,
                tgt_key_padding_mask: Optional[jt.var] = None,
                memory_key_padding_mask: Optional[jt.var] = None,
                pos_enc: Optional[jt.var] = None,
                pos_dec: Optional[jt.var] = None):

        return self.forward_post(tgt, memory, tgt_mask, memory_mask,
                                 tgt_key_padding_mask, memory_key_padding_mask, pos_enc, pos_dec)

class FeatureFusionLayer(jt.nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu"):
        super().__init__()
        self.self_attn1 = MultiheadAttention(d_model, nhead, dropout=dropout)
        self.self_attn2 = MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn1 = MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn2 = MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear11 = jt.nn.Linear(d_model, dim_feedforward)
        self.dropout1 = jt.nn.Dropout(dropout)
        self.linear12 = jt.nn.Linear(dim_feedforward, d_model)

        self.linear21 = jt.nn.Linear(d_model, dim_feedforward)
        self.dropout2 = jt.nn.Dropout(dropout)
        self.linear22 = jt.nn.Linear(dim_feedforward, d_model)

        self.norm11 = jt.nn.LayerNorm(d_model)
        self.norm12 = nn.LayerNorm(d_model)
        self.norm13 = nn.LayerNorm(d_model)
        self.norm21 = nn.LayerNorm(d_model)
        self.norm22 = nn.LayerNorm(d_model)
        self.norm23 = nn.LayerNorm(d_model)
        self.dropout11 = nn.Dropout(dropout)
        self.dropout12 = nn.Dropout(dropout)
        self.dropout13 = nn.Dropout(dropout)
        self.dropout21 = nn.Dropout(dropout)
        self.dropout22 = nn.Dropout(dropout)
        self.dropout23 = nn.Dropout(dropout)

        self.activation1 = _get_activation_fn(activation)
        self.activation2 = _get_activation_fn(activation)

    def with_pos_embed(self, tensor, pos: Optional[jt.var]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, src1, src2,
                     src1_mask: Optional[jt.var] = None,
                     src2_mask: Optional[jt.var] = None,
                     src1_key_padding_mask: Optional[jt.var] = None,
                     src2_key_padding_mask: Optional[jt.var] = None,
                     pos_src1: Optional[jt.var] = None,
                     pos_src2: Optional[jt.var] = None):

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

    def forward(self, src1, src2,
                src1_mask: Optional[jt.var] = None,
                src2_mask: Optional[jt.var] = None,
                src1_key_padding_mask: Optional[jt.var] = None,
                src2_key_padding_mask: Optional[jt.var] = None,
                pos_src1: Optional[jt.var] = None,
                pos_src2: Optional[jt.var] = None):

        return self.forward_post(src1, src2, src1_mask, src2_mask,
                                 src1_key_padding_mask, src2_key_padding_mask, pos_src1, pos_src2)


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def build_featurefusion_network(settings):
    return FeatureFusionNetwork(
        d_model=settings.hidden_dim*2,
        dropout=settings.dropout,
        nhead=settings.nheads,
        dim_feedforward=settings.dim_feedforward,
        num_featurefusion_layers=settings.featurefusion_layers
    )

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
        return nn.gelu
    if activation == "glu":
        return glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")





