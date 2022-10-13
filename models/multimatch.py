from doctest import DONT_ACCEPT_TRUE_FOR_1
import logging
from models.hcn import Skeleton
import torch
from torch import nn
from torch.nn.init import normal_, constant_
import torchvision
import numpy as np

from models.dual_builder import RGBXEncoderDecoder as dualsegmodel
from models.builder import EncoderDecoder as segmodel
from models.criterion import FixMatchLoss, MultiMatchLoss
# class labTSNMultiMatch(nn.Module):
#     def __init__(self, args):
#         super(labTSNMultiMatch, self).__init__()

#         self.l_to_ab = TSN(num_class=args.num_classes, dropout=0.0, modality='L')
#         self.ab_to_l = TSN(num_class=args.num_classes, dropout=0.0, modality='AB')

#     def forward(self, x):
#         l, ab = torch.split(x, [1, 2], dim=1)
#         # print (l.size())
#         # print (ab.size())
#         l = l.contiguous()
#         ab = ab.contiguous()
#         logits_l = self.l_to_ab(l)
#         logits_ab = self.ab_to_l(ab)
#         return logits_l, logits_ab

class rgbdMultiMatch(nn.Module):
    def __init__(self, config, norm_layer=None):
        super().__init__()

        self.l_to_ab = segmodel(cfg=config, criterion=None, norm_layer=norm_layer)
        self.ab_to_l = segmodel(cfg=config, criterion=None, norm_layer=norm_layer)

    def forward(self, l, ab):
        l = l.contiguous().clone()
        ab = ab.contiguous().clone()
        _, logits_l = self.l_to_ab(l)
        _, logits_ab = self.ab_to_l(ab)
        return logits_l, logits_ab

class rgbdFusMultiMatch(nn.Module):
    def __init__(self, config, norm_layer=None):
        super(rgbdFusMultiMatch, self).__init__()

        self.l_to_ab = segmodel(cfg=config, criterion=None, norm_layer=norm_layer)
        self.ab_to_l = segmodel(cfg=config, criterion=None, norm_layer=norm_layer)
        self.l_and_ab = dualsegmodel(cfg=config, criterion=None, norm_layer=norm_layer)

    def forward(self, l, ab):
        l = l.contiguous().clone()
        ab = ab.contiguous().clone()
        _, logits_l = self.l_to_ab(l)
        _, logits_ab = self.ab_to_l(ab)
        # print (f_l.size())
        # print (f_ab.size())
        logits_en = self.l_and_ab(l, ab) # late fusion
        return logits_l, logits_ab, logits_en