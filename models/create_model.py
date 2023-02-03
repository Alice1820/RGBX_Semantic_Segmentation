from cmath import log
import torch
from models.dual_builder import RGBXEncoderDecoder as dualsegmodel
from models.builder import EncoderDecoder as segmodel
from models.multimatch import rgbdFusMultiMatch
from models.criterion import FixMatchLoss, MultiMatchLoss

from .net_utils import net_statistics, dualnet_statistics

import logging
logger = logging.getLogger(__name__)

def create_model(config, criterion, norm_layer=None):
    if config.algo == 'supervised':
        if config.modals in ['RGB', 'Depth']:
            model = segmodel(cfg=config, criterion=criterion, norm_layer=norm_layer)
            net_statistics(model, logger)
        elif config.modals == 'RGBD':
            model = dualsegmodel(cfg=config, criterion=criterion, norm_layer=norm_layer)
            dualnet_statistics(model, logger)
    elif config.algo == 'multimatch':
        assert config.modals == 'RGBD'
        model = rgbdFusMultiMatch(config, criterion=MultiMatchLoss(use_cr=config.use_cr, threshold=config.threshold), norm_layer=norm_layer)
        # print model statistics
    elif config.algo == 'fixmatch':
        assert config.modals == 'RGBD'
        model = rgbdFusMultiMatch(config, criterion=FixMatchLoss(), norm_layer=norm_layer)
    else:
        raise NotImplementedError
    return model