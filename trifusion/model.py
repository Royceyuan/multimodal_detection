"""RGB-anchored infrared/depth residuals for the official five-scale Co-DETR."""

import math

import torch
from torch import nn
from torch.nn import functional as F

from mmdet.models.builder import DETECTORS
from projects.models import CoDETR


class ConvGnAct(nn.Sequential):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False),
            nn.GroupNorm(math.gcd(out_channels, 8), out_channels),
            nn.GELU(),
        )


class AuxiliaryPyramid(nn.Module):
    """Independent lightweight modality features at strides 4, 8, 16, 32, 64."""

    def __init__(self, in_channels, channels):
        super().__init__()
        self.stages = nn.ModuleList(
            ConvGnAct(in_channels if index == 0 else channels, channels, stride=2)
            for index in range(6)
        )

    def forward(self, x):
        outputs = []
        for index, stage in enumerate(self.stages):
            x = stage(x)
            if index >= 1:
                outputs.append(x)
        return outputs


class GatedResidual(nn.Module):
    """Zero output plus a nonzero sigmoid gate allows a first-step gradient."""

    def __init__(self, rgb_channels, aux_channels):
        super().__init__()
        self.gate = nn.Conv2d(rgb_channels + aux_channels, 1, 1)
        self.output = nn.Conv2d(aux_channels, rgb_channels, 1)
        self.init_weights()

    def init_weights(self):
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, -2.0)

    def forward(self, rgb, aux, valid=None):
        aux = F.interpolate(aux, size=rgb.shape[-2:], mode="bilinear", align_corners=False)
        gate = self.gate(torch.cat((rgb, aux), dim=1)).sigmoid()
        residual = self.output(aux) * gate
        if valid is not None:
            mask = F.interpolate(valid.to(residual.dtype), size=rgb.shape[-2:], mode="nearest")
            residual = residual * mask
        return residual.to(rgb.dtype)


class TriModalFeatureFusion(nn.Module):
    def __init__(self, fusion_channels=256, aux_channels=32, num_fusion_levels=5):
        super().__init__()
        if num_fusion_levels != 5:
            raise ValueError("Tri-modal SFP fusion requires five P2-P6 levels")
        if fusion_channels <= 0 or aux_channels <= 8:
            raise ValueError("fusion_channels must be positive and aux_channels must exceed 8")
        self.num_levels = num_fusion_levels
        self.fusion_channels = fusion_channels
        self.ir_pyramid = AuxiliaryPyramid(1, aux_channels)
        self.depth_pyramid = AuxiliaryPyramid(4, aux_channels)
        self.ir_fusions = nn.ModuleList(
            GatedResidual(fusion_channels, aux_channels) for _ in range(num_fusion_levels)
        )
        self.depth_fusions = nn.ModuleList(
            GatedResidual(fusion_channels, aux_channels) for _ in range(num_fusion_levels)
        )

    def init_weights(self):
        # MMCV BaseModule calls child init_weights after its generic initializer.
        # Reset only the new residual heads so their RGB identity survives it.
        for fusion in (*self.ir_fusions, *self.depth_fusions):
            fusion.init_weights()

    def forward(self, features, img):
        if img.ndim != 4 or img.shape[1] != 8:
            raise ValueError("Tri-modal input must be N8HW: RGB, IR, four depth channels")
        if not isinstance(features, (list, tuple)) or len(features) != self.num_levels:
            raise ValueError("Official SFP must return five feature tensors")
        valid = img[:, 6:7].clamp(0, 1)
        infrared = self.ir_pyramid(img[:, 3:4])
        depth = self.depth_pyramid(img[:, 4:8] * valid)
        outputs = []
        for index, rgb in enumerate(features):
            if rgb.ndim != 4 or rgb.shape[:2] != (img.shape[0], self.fusion_channels):
                raise ValueError("SFP feature batch/channels do not match fusion configuration")
            ir_residual = self.ir_fusions[index](rgb, infrared[index])
            depth_residual = self.depth_fusions[index](rgb, depth[index], valid)
            outputs.append(rgb + ir_residual + depth_residual)
        return tuple(outputs)


@DETECTORS.register_module()
class TriModalCoDETR(CoDETR):
    """Preserve official RGB ViT/SFP and all Co-DINO detection loss equations."""

    def __init__(self, *args, fusion_channels=256, aux_channels=32,
                 num_fusion_levels=5, fusion_enabled=True, **kwargs):
        super().__init__(*args, **kwargs)
        if not isinstance(fusion_enabled, bool):
            raise TypeError("fusion_enabled must be a boolean")
        self.fusion_enabled = fusion_enabled
        self.tri_fusion = TriModalFeatureFusion(
            fusion_channels, aux_channels, num_fusion_levels
        )

    def extract_feat(self, img, img_metas=None):
        if img.ndim != 4 or img.shape[1] != 8:
            raise ValueError("TriModalCoDETR requires the eight-channel multimodal input")
        rgb_features = super().extract_feat(img[:, :3].contiguous(), img_metas)
        return self.tri_fusion(rgb_features, img) if self.fusion_enabled else rgb_features
