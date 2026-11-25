"""Checkpoint-compatible Pan-Mamba network. Derived from the published Pan-Mamba implementation at ec1b9ead9a02b482851be98b7783111880bdd176. Parameter names and active operations are retained."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import HinResBlock, PatchEmbed, PatchUnEmbed
from .fusion import SingleMambaBlock, TokenSwapMamba, CrossMamba
from .refine import Refine
from . import _mamba_simple

class Net(nn.Module):
    def __init__(self,num_channels=None,base_filter=None,args=None):
        super(Net, self).__init__()
        base_filter=32
        self.base_filter = base_filter
        self.stride=1
        self.patch_size=1
        self.pan_encoder = nn.Sequential(nn.Conv2d(1,base_filter,3,1,1),HinResBlock(base_filter,base_filter),HinResBlock(base_filter,base_filter),HinResBlock(base_filter,base_filter))
        self.ms_encoder = nn.Sequential(nn.Conv2d(4,base_filter,3,1,1),HinResBlock(base_filter,base_filter),HinResBlock(base_filter,base_filter),HinResBlock(base_filter,base_filter))
        self.embed_dim = base_filter*self.stride*self.patch_size
        self.shallow_fusion1 = nn.Conv2d(base_filter*2,base_filter,3,1,1)
        self.shallow_fusion2 = nn.Conv2d(base_filter*2,base_filter,3,1,1)
        self.ms_to_token = PatchEmbed(in_chans=base_filter,embed_dim=self.embed_dim,patch_size=self.patch_size,stride=self.stride)
        self.pan_to_token = PatchEmbed(in_chans=base_filter,embed_dim=self.embed_dim,patch_size=self.patch_size,stride=self.stride)
        self.deep_fusion1= CrossMamba(self.embed_dim)
        self.deep_fusion2 = CrossMamba(self.embed_dim)
        self.deep_fusion3 = CrossMamba(self.embed_dim)
        self.deep_fusion4 = CrossMamba(self.embed_dim)
        self.deep_fusion5 = CrossMamba(self.embed_dim)

        self.pan_feature_extraction = nn.Sequential(*[SingleMambaBlock(self.embed_dim) for i in range(8)])
        self.ms_feature_extraction = nn.Sequential(*[SingleMambaBlock(self.embed_dim) for i in range(8)])
        self.swap_mamba1 = TokenSwapMamba(self.embed_dim)
        self.swap_mamba2 = TokenSwapMamba(self.embed_dim)
        self.patchunembe = PatchUnEmbed(base_filter)
        self.output = Refine(base_filter,4)

    def forward(self, ms, _, pan):
        # The historical variable ms_bic is actually NEAREST, not bicubic.
        if _mamba_simple.mamba_inner_fn is None or _mamba_simple.mamba_inner_fn_no_out_proj is None:
            raise RuntimeError("Pan-Mamba requires the original Vim-compatible fused kernels. See modern/README.md; do not substitute standard Mamba or disable the fast path.")
        ms_bic = F.interpolate(ms, scale_factor=4, mode="nearest")
        ms_f = self.ms_encoder(ms_bic)
        b, c, h, w = ms_f.shape
        pan_f = self.pan_encoder(pan)
        ms_f = self.ms_to_token(ms_f)
        pan_f = self.pan_to_token(pan_f)
        ms_f, residual_ms_f = self.ms_feature_extraction([ms_f, 0])
        pan_f, residual_pan_f = self.pan_feature_extraction([pan_f, 0])

        for swap in (self.swap_mamba1, self.swap_mamba2):
            ms_f, pan_f, residual_ms_f, residual_pan_f = swap(
                ms_f, pan_f, residual_ms_f, residual_pan_f
            )

        ms_f = self.patchunembe(ms_f, (h, w))
        pan_f = self.patchunembe(pan_f, (h, w))
        # Deliberately sequential: PAN fusion consumes the UPDATED MS feature.
        ms_f = self.shallow_fusion1(torch.concat([ms_f, pan_f], dim=1)) + ms_f
        pan_f = self.shallow_fusion2(torch.concat([pan_f, ms_f], dim=1)) + pan_f
        ms_f = self.ms_to_token(ms_f)
        pan_f = self.pan_to_token(pan_f)
        residual_ms_f = 0
        for fusion in (
            self.deep_fusion1, self.deep_fusion2, self.deep_fusion3,
            self.deep_fusion4, self.deep_fusion5,
        ):
            ms_f, residual_ms_f = fusion(
                ms_f, residual_ms_f, pan_f, spatial_size=(h, w)
            )

        ms_f = self.patchunembe(ms_f, (h, w))
        return self.output(ms_f) + ms_bic
