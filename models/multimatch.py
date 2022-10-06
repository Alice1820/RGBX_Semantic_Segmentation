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

class rgbxMultiMatch(nn.Module):
    def __init__(self, config, criterion, norm_layer=None):
        super().__init__()

        self.l_to_ab = segmodel(cfg=config, criterion=criterion, norm_layer=norm_layer)
        self.ab_to_l = segmodel(cfg=config, criterion=criterion, norm_layer=norm_layer)

    def forward(self, l, ab):
        l = l.contiguous().clone()
        ab = ab.contiguous().clone()
        f_l, logits_l = self.l_to_ab(l)
        f_ab, logits_ab = self.ab_to_l(ab)
        return logits_l, logits_ab

class rgbdTSNFusMMTM(nn.Module):
    def __init__(self, args):
        super(rgbdTSNFusMMTM, self).__init__()

        self.l_to_ab = TSN(num_class=args.num_classes, dropout=0.8, modality='RGB', is_shift=args.shift)
        self.ab_to_l = TSN(num_class=args.num_classes, dropout=0.8, modality='Depth', is_shift=args.shift)
        self.fc = nn.Linear(512 * 2, args.num_classes)

    def forward(self, l, ab):
        l = l.contiguous().clone()
        ab = ab.contiguous().clone()
        f_l, logits_l = self.l_to_ab(l)
        f_ab, logits_ab = self.ab_to_l(ab)
        # print (f_l.size())
        # print (f_ab.size())
        logits_en = self.fc(torch.cat([f_l, f_ab], dim=-1).contiguous()) # late fusion
        return logits_l, logits_ab, logits_en

class rgbdTSNFusMultiMatch(nn.Module):
    def __init__(self, args):
        super(rgbdTSNFusMultiMatch, self).__init__()

        self.l_to_ab = TSN(num_class=args.num_classes, dropout=0.8, modality='RGB', is_shift=args.shift)
        self.ab_to_l = TSN(num_class=args.num_classes, dropout=0.8, modality='Depth', is_shift=args.shift)
        if args.fusion_scheme == "late":
            self.l_and_ab = rgbdTSNFusLate(args)
        elif args.fusion_scheme == "early":
            self.l_and_ab = rgbdTSNFusEarly(args)
        elif args.fusion_scheme == "mmtm":
            raise Exception("Not Implemented.")
            self.l_and_ab = rgbdTSNFusMMTM(args)
        else:
            raise Exception("Not Implemented.")

    def forward(self, l, ab):
        l = l.contiguous().clone()
        ab = ab.contiguous().clone()
        _, logits_l = self.l_to_ab(l)
        _, logits_ab = self.ab_to_l(ab)
        # print (f_l.size())
        # print (f_ab.size())
        logits_en = self.l_and_ab(l, ab) # late fusion
        return logits_l, logits_ab, logits_en

class rgbsFusMultiMatch(nn.Module):
    def __init__(self, args):
        super(rgbsFusMultiMatch, self).__init__()

        self.l_to_ab = TSN(num_class=args.num_classes, dropout=0.8, modality='RGB', is_shift=args.shift)
        self.ab_to_l = Skeleton(num_classes=args.num_classes, vid_len=(8, 32))
        if args.fusion_scheme == "late":
            self.l_and_ab = rgbsFusLate(args)
        elif args.fusion_scheme == "early":
            raise Exception("Not Implemented.")
        elif args.fusion_scheme == "mmtm":
            raise Exception("Not Implemented.")
            self.l_and_ab = rgbdTSNFusMMTM(args)
        else:
            raise Exception("Not Implemented.")

    def forward(self, l, ab):
        l = l.contiguous().clone()
        ab = ab.contiguous().clone()
        _, logits_l = self.l_to_ab(l)
        _, logits_ab = self.ab_to_l(ab)
        # print (f_l.size())
        # print (f_ab.size())
        logits_en = self.l_and_ab(l, ab) # late fusion
        return logits_l, logits_ab, logits_en

class rgbsMultiMatch(nn.Module):
    def __init__(self, args):
        super(rgbsMultiMatch, self).__init__()

        if args.backbone_vis == "TSN":
            self.l_to_ab = TSN(num_class=args.num_classes, dropout=0.8, modality='RGB', is_shift=args.shift)
        elif args.backbone_vis == "I3D":
            self.l_to_ab = I3D(num_class=args.num_classes, dropout=0.8, vid_len=8)
        else:
            raise NotImplementedError
        self.ab_to_l = Skeleton(num_classes=args.num_classes, vid_len=(8, 32))

    def forward(self, l, ab):
        l = l.contiguous()
        ab = ab.contiguous()
        f_l, logits_l = self.l_to_ab(l)
        f_ab, logits_ab = self.ab_to_l(ab)
        return logits_l, logits_ab

class cifarMultiMatch(nn.Module):
    def __init__(self, args, ensemble=False):
        super(cifarMultiMatch, self).__init__()
        
        logger = logging.getLogger(__name__)
        self.ensemble = ensemble

        if args.arch == 'wideresnet':
            import models.wideresnet as models
            self.l_to_ab = models.build_wideresnet(
                                            in_channels=1,
                                            depth=args.model_depth,
                                            widen_factor=args.model_width,
                                            dropout=0,
                                            num_classes=args.num_classes)
            self.ab_to_l = models.build_wideresnet(
                                            in_channels=2,
                                            depth=args.model_depth,
                                            widen_factor=args.model_width,
                                            dropout=0,
                                            num_classes=args.num_classes)
        elif args.arch == 'resnext':
            import models.resnext as models
            self.l_to_ab = models.build_resnext(
                                        in_channels=1,
                                         cardinality=args.model_cardinality,
                                         depth=args.model_depth,
                                         width=args.model_width,
                                         num_classes=args.num_classes)
            self.l_to_ab = models.build_resnext(
                                        in_channels=2,
                                         cardinality=args.model_cardinality,
                                         depth=args.model_depth,
                                         width=args.model_width,
                                         num_classes=args.num_classes)
        logger.info("Total params: {:.2f}M {:.2f}M".format(
            sum(p.numel() for p in self.l_to_ab.parameters())/1e6,
            sum(p.numel() for p in self.ab_to_l.parameters())/1e6))
        
        if self.ensemble:
            self.fc = nn.Linear(self.l_to_ab.channels + self.ab_to_l.channels, args.num_classes)


    def forward(self, x):
        l, ab = torch.split(x, [1, 2], dim=1)
        # print (l.size())
        # print (ab.size())
        l = l.contiguous()
        ab = ab.contiguous()
        f_l, logits_l = self.l_to_ab(l)
        f_ab, logits_ab = self.ab_to_l(ab)
        # print (f_l.size(), f_ab.size()) # [512, 120] [512, 120]
        if self.ensemble:
            logits_en = self.fc(torch.cat([f_l.detach(), f_ab.detach()], dim=-1))
            return logits_l, logits_ab, logits_en
        else:
            return logits_l, logits_ab