import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import warnings
from torch.nn.modules.utils import _pair as to_2tuple
from mmseg.models.builder import BACKBONES
from mmseg.utils import get_root_logger
from .mmcv_custom import load_checkpoint
from .involution_cuda import involution
from mmcv.cnn import (ConvModule, build_conv_layer, build_norm_layer,
                      constant_init, kaiming_init)
from mmcv.runner import BaseModule
from mmcv.cnn.bricks import DropPath
from mmcv.cnn.utils.weight_init import (constant_init, normal_init,
                                        trunc_normal_init)


class Mlp(BaseModule):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)

        x = self.dwconv(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)

        return x


class StemConv(BaseModule):
    def __init__(self, in_channels, out_channels, norm_cfg=dict(type='SyncBN', requires_grad=True)):
        super(StemConv, self).__init__()

        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 2,
                      kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            build_norm_layer(norm_cfg, out_channels // 2)[1],
            nn.GELU(),
            nn.Conv2d(out_channels // 2, out_channels,
                      kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            build_norm_layer(norm_cfg, out_channels)[1],
        )

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.size()
        x = x.flatten(2).transpose(1, 2)
        return x, H, W


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x):
        x = self.dwconv(x)
        return x



class CrossModalAttention(nn.Module):
    def __init__(self, v_in_channels, l_in_channels, key_channels, value_channels, out_channels=None, num_heads=1):
        super(CrossModalAttention, self).__init__()

        self.v_in_channels = v_in_channels
        self.l_in_channels = l_in_channels
        self.out_channels = out_channels
        self.key_channels = key_channels
        self.value_channels = value_channels
        self.num_heads = num_heads
        
        if out_channels is None:
            self.out_channels = self.value_channels

        # Keys: language features: (B, l_in_channels, #words)
        self.project_k = nn.Sequential(
            nn.Conv1d(self.l_in_channels, self.key_channels, kernel_size=1, stride=1),
        )

        # Queries: visual features: (B, H*W, v_in_channels)
        self.project_q = nn.Sequential(
            nn.Conv1d(self.v_in_channels, self.key_channels, kernel_size=1, stride=1),
            nn.InstanceNorm1d(self.key_channels),
        )

        # Values: language features: (B, l_in_channels, #words)
        self.project_v = nn.Sequential(
            nn.Conv1d(self.l_in_channels, self.value_channels, kernel_size=1, stride=1),
        )

        # Out projection
        self.W = nn.Sequential(
            nn.Conv1d(self.value_channels, self.out_channels, kernel_size=1, stride=1),
            nn.InstanceNorm1d(self.out_channels),
        )

    def forward(self, x, l, l_mask):
        # x shape: (B, H*W, C_v)
        # l input shape: (B, C_l, T)
        # l_mask shape: (B, T, 1)
        
        B, HW = x.size(0), x.size(1)
        x = x.permute(0, 2, 1)  # (B, key_channels, H*W)
        l_mask = l_mask.permute(0, 2, 1)  # (B, T, 1) -> (B, 1, T)

        query = self.project_q(x)  # (B, key_channels, H*W) 
        query = query.permute(0, 2, 1)  # (B, H*W, key_channels)
        key = self.project_k(l)  # (B, key_channels, T)
        value = self.project_v(l)  # (B, value_channels, T)

        key = key * l_mask  # (B, key_channels, T)
        value = value * l_mask  # (B, value_channels, T)
        n_l = value.size(-1)
        query = query.reshape(B, HW, self.num_heads, self.key_channels//self.num_heads).permute(0, 2, 1, 3)
        # (B, num_heads, H*W, key_channels//num_heads)
        key = key.reshape(B, self.num_heads, self.key_channels//self.num_heads, n_l)
        # (B, num_heads, key_channels//num_heads, T)
        value = value.reshape(B, self.num_heads, self.value_channels//self.num_heads, n_l)
        # # (B, num_heads, value_channels//num_heads, T)
        l_mask = l_mask.unsqueeze(1)  # (B, 1, 1, T)

        # attention score
        sim_map = torch.matmul(query, key)  # (B, self.num_heads, H*W, T)
        sim_map = (self.key_channels ** -.5) * sim_map  # scaled dot product

        sim_map = sim_map + (1e4*l_mask - 1e4)  # assign a very small number to padding positions
        sim_map = F.softmax(sim_map, dim=-1)  # (B, num_heads, h*w, T)

        # visual-linguistic correlation
        sim_temp = torch.sum(sim_map, dim=3)
        
        out = torch.matmul(sim_map, value.permute(0, 1, 3, 2))  # (B, num_heads, H*W, value_channels//num_heads)
        out = out.permute(0, 2, 1, 3).contiguous().reshape(B, HW, self.value_channels)  # (B, H*W, value_channels)
        out = out.permute(0, 2, 1)  # (B, value_channels, HW)
        out = self.W(out)  # (B, out_channels, HW) 
        out = out.permute(0, 2, 1)  # (B, HW, out_channels)

        return out, sim_temp


class GatedCrossModalAttention(nn.Module):
    def __init__(self, dim, v_in_channels, l_in_channels, key_channels, value_channels, num_heads=0, dropout=0.0):
        super(GatedCrossModalAttention, self).__init__()
        
        self.project_v = nn.Sequential(nn.Conv1d(dim, dim, 1, 1),  
                                       nn.GELU(),
                                       nn.Dropout(dropout)
                                      )

        self.vis_lang_att = CrossModalAttention(v_in_channels,  # v_in
                                                l_in_channels,  # l_in
                                                key_channels,  # key
                                                value_channels,  # value
                                                out_channels=value_channels,  # out
                                                num_heads=num_heads
                                               )

        self.res_gate = nn.Sequential(
            nn.Linear(dim, dim, bias=False),
            nn.ReLU(),
            nn.Linear(dim, dim, bias=False),
            nn.Tanh()
        )

        # self.res_gate = nn.Sequential(
        #     nn.Linear(dim, dim, bias=False),
        #     nn.GELU()
        # )


    def forward(self, x, l, l_mask):
        # input x shape: (B, H*W, C_v)

        cross, sim_temp = self.vis_lang_att(x, l, l_mask)  # (B, H*W, C_v)

        gated_cross = self.res_gate(cross) * cross

        return gated_cross, sim_temp

class LMFA(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)

        self.conv0_1 = nn.Conv2d(dim, dim, (1, 7), padding=(0, 3), groups=dim)
        self.conv0_2 = nn.Conv2d(dim, dim, (7, 1), padding=(3, 0), groups=dim)

        self.conv1_1 = nn.Conv2d(dim, dim, (1, 11), padding=(0, 5), groups=dim)
        self.conv1_2 = nn.Conv2d(dim, dim, (11, 1), padding=(5, 0), groups=dim)

        self.conv2_1 = nn.Conv2d(dim, dim, (1, 21), padding=(0, 10), groups=dim)
        self.conv2_2 = nn.Conv2d(dim, dim, (21, 1), padding=(10, 0), groups=dim)
        self.conv3 = nn.Conv2d(dim, dim, 1)

        self.fusion = GatedCrossModalAttention(dim, dim, 768, dim, dim, num_heads=1, dropout=0.)
        

    def forward(self, x, l, l_mask):

        # visual input shape: (B, C_v, H, W)
        # linguistic input shape: (B, C_l, T)
        
        u = x.clone()
        attn = self.conv0(x)
        B, _, H, W = attn.shape

        # Convolutional Branches
        attn_0 = self.conv0_1(attn)
        attn_0 = self.conv0_2(attn_0)

        attn_1 = self.conv1_1(attn)
        attn_1 = self.conv1_2(attn_1)
        
        attn_2 = self.conv2_1(attn)
        attn_2 = self.conv2_2(attn_2)

        # Cross-Modal Branch
        # visual input shape is changed to (B, H*W, C_v)
        attn_3, sim_temp = self.fusion(attn.reshape(B, -1, H*W).permute(0, 2, 1).contiguous(), l, l_mask) 
        attn_3 = attn_3.permute(0, 2, 1).contiguous().reshape(B, -1, H, W) # shape reverse (B, C_v, H, W)

        # Integrated Attention
        attn = self.conv3(attn + attn_0 + attn_1 + attn_2 + attn_3)

        out = attn * u
        
        return out, sim_temp


class SpatialAttention(BaseModule):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        self.proj_1 = nn.Conv2d(d_model, d_model, 1)
        self.activation = nn.GELU()
        self.spatial_gating_unit = LMFA(d_model)
        self.proj_2 = nn.Conv2d(d_model, d_model, 1)
        self.depths=[3, 3, 12, 3]


    def forward(self, x, l, l_mask, i, j):
        shorcut = x.clone()
        B, _, H, W = x.shape
        x = self.proj_1(x)
        x = self.activation(x)

        x, sim_temp = self.spatial_gating_unit(x, l, l_mask)
        x = self.proj_2(x)
        x = x + shorcut
        return x, sim_temp


class Block(BaseModule):

    def __init__(self,
                 dim,
                 mlp_ratio=4.,
                 drop=0.,
                 drop_path=0.,
                 act_layer=nn.GELU,
                 norm_cfg=dict(type='SyncBN', requires_grad=True)):
        super().__init__()
        self.norm1 = build_norm_layer(norm_cfg, dim)[1]
        self.attn = SpatialAttention(dim)
        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = build_norm_layer(norm_cfg, dim)[1]
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
                       act_layer=act_layer, drop=drop)
        layer_scale_init_value = 1e-2
        self.layer_scale_1 = nn.Parameter(
            layer_scale_init_value * torch.ones((dim)), requires_grad=True)
        self.layer_scale_2 = nn.Parameter(
            layer_scale_init_value * torch.ones((dim)), requires_grad=True)

    def forward(self, x, H, W, l, l_mask, i, j):
        B, N, C = x.shape
        x = x.permute(0, 2, 1).view(B, C, H, W)
        att, sim_temp = self.attn(self.norm1(x), l, l_mask, i, j)
        x = x + self.drop_path(self.layer_scale_1.unsqueeze(-1).unsqueeze(-1) * att)
        
        x = x + self.drop_path(self.layer_scale_2.unsqueeze(-1).unsqueeze(-1) * self.mlp(self.norm2(x)))
        
        
        x = x.view(B, C, N).permute(0, 2, 1)
        return x, sim_temp


class OverlapPatchEmbed(BaseModule):
    """ Image to Patch Embedding
    """

    def __init__(self, patch_size=7, stride=4, in_chans=3, embed_dim=768, norm_cfg=dict(type='SyncBN', requires_grad=True)):
        super().__init__()
        patch_size = to_2tuple(patch_size)

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))
        self.norm = build_norm_layer(norm_cfg, embed_dim)[1]

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = self.norm(x)

        x = x.flatten(2).transpose(1, 2)

        return x, H, W


@ BACKBONES.register_module()
class SLViT_Encoder(BaseModule):
    def __init__(self,
                 in_chans=3,
                 embed_dims=[64, 128, 256, 512],
                 mlp_ratios=[4, 4, 4, 4],
                 drop_rate=0.,
                 drop_path_rate=0.,
                 depths=[3, 4, 6, 3],
                 num_stages=4,
                 norm_cfg=dict(type='SyncBN', requires_grad=True),
                 pretrained=None,
                 init_cfg=None):
        super(SLViT_Encoder, self).__init__(init_cfg=init_cfg)

        assert not (init_cfg and pretrained), \
            'init_cfg and pretrained cannot be set at the same time'
        if isinstance(pretrained, str):
            warnings.warn('DeprecationWarning: pretrained is deprecated, '
                          'please use "init_cfg" instead')
            self.init_cfg = dict(type='Pretrained', checkpoint=pretrained)
        elif pretrained is not None:
            raise TypeError('pretrained must be a str or None')

        self.depths = depths
        self.num_stages = num_stages

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate,
                                                sum(depths))]  # stochastic depth decay rule
        cur = 0
        self.topks = 32

        for i in range(num_stages):
            if i == 0:
                patch_embed = StemConv(3, embed_dims[0], norm_cfg=norm_cfg)
            else:
                patch_embed = OverlapPatchEmbed(patch_size=7 if i == 0 else 3,
                                                stride=4 if i == 0 else 2,
                                                in_chans=in_chans if i == 0 else embed_dims[i - 1],
                                                embed_dim=embed_dims[i],
                                                norm_cfg=norm_cfg)

            block = nn.ModuleList([Block(dim=embed_dims[i], mlp_ratio=mlp_ratios[i],
                                         drop=drop_rate, drop_path=dpr[cur + j],
                                         norm_cfg=norm_cfg)
                                   for j in range(depths[i])])
            norm = nn.LayerNorm(embed_dims[i])

            cur += depths[i]

            setattr(self, f"patch_embed{i + 1}", patch_embed)
            setattr(self, f"block{i + 1}", block)
            setattr(self, f"norm{i + 1}", norm)


    def init_weights(self, pretrained=None):
        """Initialize the weights in backbone.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """
        # print('init cfg', self.init_cfg)
        def _init_weights(m):
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    trunc_normal_init(m, std=.02, bias=0.)
                elif isinstance(m, nn.LayerNorm):
                    constant_init(m, val=1.0, bias=0.)
                elif isinstance(m, nn.Conv2d):
                    fan_out = m.kernel_size[0] * m.kernel_size[
                        1] * m.out_channels
                    fan_out //= m.groups
                    normal_init(
                        m, mean=0, std=math.sqrt(2.0 / fan_out), bias=0)

        if isinstance(pretrained, str):
            self.apply(_init_weights)
            logger = get_root_logger()
            # load_checkpoint(self, pretrained, strict=True, logger=logger)
            load_checkpoint(self, pretrained, strict=('upernet' in pretrained), logger=logger)
        elif pretrained is None:
            self.apply(_init_weights)
        else:
            raise TypeError('pretrained must be a str or None')

    def _init_weights(self, m):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                trunc_normal_init(m, std=.02, bias=0.)
            elif isinstance(m, nn.LayerNorm):
                constant_init(m, val=1.0, bias=0.)
            elif isinstance(m, nn.Conv2d):
                fan_out = m.kernel_size[0] * m.kernel_size[
                    1] * m.out_channels
                fan_out //= m.groups
                normal_init(
                    m, mean=0, std=math.sqrt(2.0 / fan_out), bias=0)

    def forward(self, x, l, l_mask):
        B = x.shape[0]
        outs = []
        sim_temps = []

        for i in range(self.num_stages):
            patch_embed = getattr(self, f"patch_embed{i + 1}")
            block = getattr(self, f"block{i + 1}")
            norm = getattr(self, f"norm{i + 1}")
            x, H, W = patch_embed(x)
            j = 0
            for blk in block:
                x, sim_temp = blk(x, H, W, l, l_mask, i, j)
                j = j + 1
            x = norm(x)
            
            x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
            outs.append(x)
            sim_temps.append(sim_temp)

        return outs, sim_temps

