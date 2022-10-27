from torch import nn

from .dual_builder import RGBXEncoderDecoder as dualsegmodel
from .builder import EncoderDecoder as segmodel
from .criterion import FixMatchLoss, MultiMatchLoss

def interleave(x, size):
    s = list(x.shape)
    return x.reshape([-1, size] + s[1:]).transpose(0, 1).reshape([-1] + s[1:])


def de_interleave(x, size):
    s = list(x.shape)
    return x.reshape([size, -1] + s[1:]).transpose(0, 1).reshape([-1] + s[1:])

class rgbdMultiMatch(nn.Module):
    def __init__(self, config, criterion, norm_layer=None):
        super().__init__()

        self.l_to_ab = segmodel(cfg=config, criterion=None, norm_layer=norm_layer)
        self.ab_to_l = segmodel(cfg=config, criterion=None, norm_layer=norm_layer)

    def forward(self, l, ab):
        l = l.contiguous().clone()
        ab = ab.contiguous().clone()
        logits_l = self.l_to_ab(l)
        logits_ab = self.ab_to_l(ab)
        return logits_l, logits_ab

class rgbdFusMultiMatch(nn.Module):
    def __init__(self, config, criterion, norm_layer=None):
        super(rgbdFusMultiMatch, self).__init__()

        self.l_to_ab = segmodel(cfg=config, criterion=None, norm_layer=norm_layer)
        self.ab_to_l = segmodel(cfg=config, criterion=None, norm_layer=norm_layer)
        self.l_and_ab = dualsegmodel(cfg=config, criterion=None, norm_layer=norm_layer)
        self.config = config
        self.criterion_x = nn.CrossEntropyLoss(reduction='mean', ignore_index=config.background)
        self.criterion_u = criterion

    def forward(self, l, ab, gts=None, tb=None):
        l = l.contiguous().clone()
        ab = ab.contiguous().clone()
        logits_l = self.l_to_ab(l)['logits']
        logits_ab = self.ab_to_l(ab)['logits']
        # print (f_l.size())
        # print (f_ab.size())
        logits_en = self.l_and_ab(l, ab)['logits'] # late fusion
        # resolve logits
        batch_size = self.config.batch_size
        logits_l = de_interleave(logits_l, self.config.mu+1)
        logits_l_x = logits_l[:batch_size]
        logits_l_u = logits_l[batch_size:]

        logits_ab = de_interleave(logits_ab, self.config.mu+1)
        logits_ab_x = logits_ab[:batch_size]
        logits_ab_u = logits_ab[batch_size:]

        logits_en = de_interleave(logits_en, self.config.mu+1)
        logits_en_x = logits_en[:batch_size]
        logits_en_u = logits_en[batch_size:]

        loss_x = self.criterion_x(logits_l_x, gts.long()) + \
                 self.criterion_x(logits_ab_x, gts.long()) + \
                 self.criterion_x(logits_en_x, gts.long())
        mask_l, mask_ab, thres_l, thres_ab, _, _, loss_lab_u = self.criterion_u(logits_l_u, logits_l_u, logits_ab_u, logits_ab_u)
        _, mask_en, _, thres_en, _, _, loss_len_u = self.criterion_u(logits_l_u, logits_l_u, logits_en_u, logits_en_u)
        _, _, _, _, _, _, loss_aben_u = self.criterion_u(logits_ab_u, logits_ab_u, logits_en_u, logits_en_u)

        # unsupervised loss
        loss_u = loss_lab_u + loss_len_u + loss_aben_u
        loss = loss_x + loss_u * self.config.lambda_u

        if tb is not None:
            pass 
        outputs = {'loss': loss, 'mask_rgb': mask_l, 'mask_dep': mask_ab, 'mask_en': mask_en, 
                   'loss_x': loss_x, 'loss_u': loss_u, 'thres_rgb': thres_l, 'thres_dep': thres_ab, 'thres_en': thres_en}
        return outputs
