from collections import OrderedDict
import sys
import torch
from torch import nn
from torch.nn import functional as F
from bert.modeling_bert import BertModel


class SpatialCrossScaleAtten(nn.Module):
    def __init__(self, dim):
        super(SpatialCrossScaleAtten, self).__init__()
        self.qkv_linear = nn.Linear(dim, dim * 3, bias=True)
        self.softmax = nn.Softmax(dim=-1)
        self.proj = nn.Linear(dim, dim)
        self.num_head = 8
        self.scale = (dim // self.num_head)**0.5

    def forward(self, x):
        B, num_blocks, _, C = x.shape  # (B, K, N, C)

        # (3, B, K, head, N, C)
        qkv = self.qkv_linear(x).reshape(B, num_blocks, -1, 3, self.num_head, C // self.num_head).permute(3, 0, 1, 4, 2, 5).contiguous() 
        q, k, v = qkv[0], qkv[1], qkv[2]

        atten = q @ k.transpose(-1, -2).contiguous()
        atten = self.softmax(atten)
        
        atten_value = (atten @ v).transpose(-2, -3).contiguous().reshape(B, num_blocks, -1, C)
        atten_value = self.proj(atten_value)  # (B, K, K, N, C) / (B, K, N, C)
        
        return atten_value

class CrossScaleEnhance(nn.Module):
    def __init__(self, dim):
        super(CrossScaleEnhance, self).__init__()
        self.Attention = SpatialCrossScaleAtten(dim)
        layer_scale_init_value = 1
        self.layer_scale = nn.Parameter(
            layer_scale_init_value * torch.ones((dim)), requires_grad=True)

    def forward(self, x):
        h = x  # (B, N, H)
        x = self.Attention(x) 
        x = h + self.layer_scale * x

        return x

class URCE(nn.Module):
    def __init__(self, dim=256, num=1):  # dim = 256
        super(URCE, self).__init__()
        self.ini_win_size = 2
        self.channels = [64, 128, 320, 512]
        self.dim = dim
        self.fc_module = nn.ModuleList()
        self.fc_rever_module = nn.ModuleList()
        self.num = num
        self.num_stages = 4
        self.topK = 32
        
        for i in range(self.num_stages):
            self.fc_module.append(nn.Linear(self.channels[i], self.dim))

        for i in range(self.num_stages):
            self.fc_rever_module.append(nn.Linear(self.dim, self.channels[i]))

        self.group_attention = []
        for i in range(self.num):
            self.group_attention.append(CrossScaleEnhance(dim))
        self.group_attention = nn.Sequential(*self.group_attention)
            
        self.split_list = [8 * 8, 4 * 4, 2 * 2, 1 * 1]
        

    def forward(self, x, sim_temp):

        # Uncertain Region Extraction
        s1, s2, s3, s4 = sim_temp
        x1, x2, x3, x4 = x
        B, C, H, W = x1.shape
        
        h = H // (self.ini_win_size ** (self.num_stages - 1))
        w = W // (self.ini_win_size ** (self.num_stages - 1))
        map_U = torch.zeros((B, 1, h, w), device=x1.device)

        s1 = s1.reshape(B, 1, H // (self.ini_win_size ** 0), W // (self.ini_win_size ** 0))
        s2 = s2.reshape(B, 1, H // (self.ini_win_size ** 1), W // (self.ini_win_size ** 1))
        s3 = s3.reshape(B, 1, H // (self.ini_win_size ** 2), W // (self.ini_win_size ** 2))
        s4 = s4.reshape(B, 1, H // (self.ini_win_size ** 3), W // (self.ini_win_size ** 3))
        s1 = F.interpolate(input=s1, size=(h, w), mode='bilinear', align_corners=True)
        s2 = F.interpolate(input=s2, size=(h, w), mode='bilinear', align_corners=True)
        s3 = F.interpolate(input=s3, size=(h ,w), mode='bilinear', align_corners=True)

        map_U = (s1 - s2).abs() + (s2 - s3).abs() + (s3 - s4).abs()

        topk_score, topk_idx = torch.topk(map_U.reshape(B,1,h*w), dim=-1, k=self.topK, largest=True)
        
        # Cross-Scale Fusing Attention
        # channel unify
        x = [self.fc_module[i](item.permute(0, 2, 3, 1)) for i, item in enumerate(x)]  # [(B, H_i, W_i, dim)]

        # patch merging
        N = 0
        for j, item in enumerate(x):
            B, H, W, C = item.shape
            win_size = self.ini_win_size ** (self.num_stages - j - 1)
            item = item.reshape(B, H // win_size, win_size, W // win_size, win_size, C).permute(0, 1, 3, 2, 4, 5).contiguous()
            item = item.reshape(B, H // win_size, W // win_size, win_size * win_size, C).contiguous()
            N = N + win_size * win_size
            x[j] = item

        x = tuple(x)
        x = torch.cat(x, dim=-2)  # (B, h, w, N, dim)
        y = torch.zeros((B, self.topK, N, self.dim), device=x.device) # (B, K, N, dim)
        
        # y: cross-scale features of uncertain regions 
        for j in range(self.topK):
            for i, item in enumerate(topk_idx):
                it = item[0][j]
                xt, yt = it // h, it % w
                y[i][j] = x[i][xt][yt]

        # multi-head self attention with spatial correspondence
        for i in range(self.num):
            y = self.group_attention[i](y)  # (B, K, N, dim)
            
        x = 2*x
        for i, item in enumerate(topk_idx):
            for j in range(self.topK):
                it = item[0][j]
                xt, yt = it // h, it % w
                x[i][xt][yt] = y[i][j]
        
        # patch reversion
        x = torch.split(x, self.split_list, dim=-2)
        x = list(x) 

        for j, item in enumerate(x):
            B, num_blocks, _, N, C = item.shape
            win_size = self.ini_win_size ** (self.num_stages - j - 1)
            item = item.reshape(B, num_blocks, num_blocks, win_size, win_size, C).permute(0, 1, 3, 2, 4, 5).contiguous().reshape(B, num_blocks*win_size, num_blocks*win_size, C)
            item = self.fc_rever_module[j](item).permute(0, 3, 1, 2).contiguous()
            x[j] = item

        return x


#####################################################################################
# SLViT: [integrated vision-language encoder] - [cross-scale enhancement] - decoder #
#####################################################################################
class _SLViT(nn.Module):
    def __init__(self, backbone, classifier, args):
        super(_SLViT, self).__init__()
        self.backbone = backbone
        self.classifier = classifier
        self.text_encoder = BertModel.from_pretrained(args.ck_bert)
        self.text_encoder.pooler = None
        self.inter_trans = URCE(dim=256)
        self.squeelayers = nn.ModuleList()
        self.fusion_list = [64, 128, 320, 512]
        self.num_module = 4
        
        self.normlayers = nn.ModuleList()
        self.act = nn.ReLU(inplace=True)
        for i in range(self.num_module):
            self.squeelayers.append(
                nn.Conv2d(self.fusion_list[i]*2, self.fusion_list[i], 1, 1)
            )
            self.normlayers.append(
                nn.LayerNorm(self.fusion_list[i])
            )

        

    def forward(self, x, text, l_mask):
        input_shape = x.shape[-2:]
        ### language inference ###
        l_feats = self.text_encoder(text, attention_mask=l_mask)[0]  # (6, 10, 768)
        l_feats = l_feats.permute(0, 2, 1)  # (B, 768, N_l) to make Conv1d happy
        l_mask = l_mask.unsqueeze(dim=-1)  # (batch, N_l, 1)

        # integrated vision-language encoder
        features, sim_temps = self.backbone(x, l_feats, l_mask)
        
        # cross-scale enhancement
        feature_trans = self.inter_trans(features, sim_temps)
        feature_inter = []
        for i in range(self.num_module):
            skip = self.squeelayers[i](torch.cat((feature_trans[i], features[i]), dim=1))
            skip = self.act(self.normlayers[i](skip.permute(0, 2, 3, 1).contiguous()).permute(0, 3, 1, 2).contiguous())
            feature_inter.append(skip)
        x_c1, x_c2, x_c3, x_c4 = feature_inter

        # decoder
        x = self.classifier(x_c4, x_c3, x_c2, x_c1, l_feats)
        x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=True)
        return x


class SLViT(_SLViT):
    pass
