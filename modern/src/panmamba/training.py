"""Opt-in paired-data trainer, retaining the released default objective and schedule."""
import torch
from torch.nn import functional as F


def reconstruction_loss(pred, target, configured_batch_size: int):
    # The legacy solver divides by configured batch size, not actual final batch size.
    return F.l1_loss(pred, target, reduction="mean") / (configured_batch_size * 2)


def make_optimizer(model, lr=5e-4):
    return torch.optim.Adam(model.parameters(), lr=lr, betas=(.9,.999), eps=1e-8)


def make_scheduler(optimizer):
    return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 500, 1, 5e-8)
