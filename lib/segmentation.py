import torch
import torch.nn as nn
from .mask_predictor import LightHamHead
from .encoder import SLViT_Encoder
from ._utils import SLViT
from functools import partial


__all__ = ['slvit']


#####################################################################################
# SLViT: [integrated vision-language encoder] - [cross-scale enhancement] - decoder #
#####################################################################################
def _segm_slvit(pretrained, args):

    # args.window12 added for test.py because state_dict is loaded after model initialization
    if 'window12' in pretrained or args.window12:
        print('Window size 12!')
        window_size = 12
    else:
        window_size = 7

    if args.mha:
        mha = args.mha.split('-')  # if non-empty, then ['a', 'b', 'c', 'd']
        mha = [int(a) for a in mha]
    else:
        mha = [1, 1, 1, 1]

    out_indices = (0, 1, 2, 3)

    backbone = SLViT_Encoder(
                     embed_dims=[64, 128, 320, 512],
                     mlp_ratios=[8, 8, 4, 4],
                     drop_rate=0.0,
                     depths=[3, 3, 12, 3],
                     drop_path_rate=0.2,
                     norm_cfg=dict(type='SyncBN', requires_grad=True))
    if pretrained:
        print('Initializing Multi-modal Transformer weights from ' + pretrained)
        backbone.init_weights(pretrained=pretrained)
    else:
        print('Randomly initialize Multi-modal Transformer weights.')
        backbone.init_weights()

    model_map = [LightHamHead, SLViT]
    classifier = model_map[0](
                ham_channels=512,
                dropout_ratio=0.1,
                ham_norm_cfg = dict(type='GN', num_groups=32, requires_grad=True))

    base_model = model_map[1]

    model = base_model(backbone, classifier, args)
    return model


def _load_model_slvit(pretrained, args):
    model = _segm_slvit(pretrained, args)
    return model


def slvit(pretrained='', args=None):
    return _load_model_slvit(pretrained, args)
