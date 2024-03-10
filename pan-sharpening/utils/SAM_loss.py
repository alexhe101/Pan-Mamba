import torch
from torch import nn
import torch.nn.functional as F

_reduction_modes = ['none', 'mean', 'sum']


def _sam(img1, img2):
    inner_product = torch.sum(img1 * img2, axis=1)
    img1_spectral_norm = torch.sqrt(torch.sum(img1 ** 2, axis=1))
    img2_spectral_norm = torch.sqrt(torch.sum(img2 ** 2, axis=1))
    # numerical stability
    cos_theta = torch.clip(inner_product / (img1_spectral_norm * img2_spectral_norm + 1e-6), min=0, max=1)
    return torch.mean(torch.acos(cos_theta))


class SAMLoss(torch.nn.Module):
    def __init__(self):
        super(SAMLoss, self).__init__()

    def forward(self, input, label):
        return _sam(input, label)


def l1_loss(pred, target):
    return F.l1_loss(pred, target, reduction='none')


class PhaLoss(nn.Module):
    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(PhaLoss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, **kwargs):
        pred_fft = torch.fft.rfft2(pred + 1e-8, norm='backward')
        pred_pha = torch.angle(pred_fft)
        target_fft = torch.fft.rfft2(target + 1e-8, norm='backward')
        target_pha = torch.angle(target_fft)
        return self.loss_weight * l1_loss(pred_pha, target_pha)

