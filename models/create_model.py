import torch
from models.dual_builder import RGBXEncoderDecoder as dualsegmodel
from models.builder import EncoderDecoder as segmodel
from models.multimatch import rgbdFusMultiMatch
from models.criterion import FixMatchLoss, MultiMatchLoss

import logging
logger = logging.getLogger(__name__)

def create_model(config, criterion, norm_layer=None):
    if config.algo == 'supervised':
        if config.modals in ['RGB', 'Depth']:
            model = segmodel(cfg=config, criterion=criterion, norm_layer=norm_layer)
        elif config.modals == 'RGBD':
            model = dualsegmodel(cfg=config, criterion=criterion, norm_layer=norm_layer)
    elif config.algo == 'multimatch':
        assert config.modals == 'RGBD'
        model = rgbdFusMultiMatch(config, criterion=MultiMatchLoss(use_cr=config.use_cr, threshold=config.threshold), norm_layer=norm_layer)
        # print model statistics
        from thop import profile
        inputs = torch.randn(1, 3, 480, 640)
        # gts = torch.zeros(1, 480, 640)
        flops, params = profile(model.l_to_ab, inputs=(inputs, ))
        logger.info('FLOPs = ' + str(flops/1000**3) + 'G')
        logger.info('Params = ' + str(params/1000**2) + 'M')
    elif config.algo == 'fixmatch':
        assert config.modals == 'RGBD'
        model = rgbdFusMultiMatch(config, criterion=FixMatchLoss(), norm_layer=norm_layer)
    else:
        raise NotImplementedError

    return model