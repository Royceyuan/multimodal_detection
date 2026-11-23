"""Independent auxiliary routes over the unchanged five-scale detector topology."""
from contextlib import nullcontext
import torch

from mmdet.models.builder import DETECTORS
from trifusion.model import CoDETR, TriModalCoDETR


def normalize_modalities(value):
    if not isinstance(value, (tuple, list)):
        raise TypeError("active_modalities must be a list or tuple")
    if any(name not in ("ir", "depth") for name in value) or len(set(value)) != len(value):
        raise ValueError("active_modalities must contain unique ir/depth names")
    return tuple(name for name in ("ir", "depth") if name in value)


@DETECTORS.register_module()
class ProgressiveCoDETR(TriModalCoDETR):
    """Keep all original checkpoint keys; omit disabled branches from the graph.

    The stage controller owns trainability of enabled routes. Disabled routes
    are always frozen. Evaluation route changes do not enable training of a
    newly selected route; the controller must apply its phase policy afterwards.
    """

    def __init__(self, *args, active_modalities=None, fusion_enabled=None, **kwargs):
        if fusion_enabled is not None and not isinstance(fusion_enabled, bool):
            raise TypeError("fusion_enabled must be a boolean")
        legacy = () if fusion_enabled is False else ("ir", "depth")
        active = legacy if active_modalities is None else normalize_modalities(active_modalities)
        if active_modalities is not None and fusion_enabled is not None and active != legacy:
            raise ValueError("fusion_enabled conflicts with active_modalities; use one interface")
        super().__init__(*args, fusion_enabled=bool(active), **kwargs)
        self._active_modalities = active
        self._freeze_disabled()

    @property
    def active_modalities(self):
        return self._active_modalities

    def _branch_modules(self, name):
        return (getattr(self.tri_fusion, name + "_pyramid"),
                getattr(self.tri_fusion, name + "_fusions"))

    def _freeze_disabled(self):
        for name in ("ir", "depth"):
            if name not in self.active_modalities:
                for module in self._branch_modules(name):
                    module.eval()
                    for parameter in module.parameters():
                        parameter.requires_grad_(False)
                        parameter.grad = None

    def set_active_modalities(self, active_modalities):
        """Select an evaluation ablation; training route changes require a new stage."""
        if self.training:
            raise RuntimeError("Change modality routes only in eval mode")
        self._active_modalities = normalize_modalities(active_modalities)
        self.fusion_enabled = bool(self.active_modalities)
        self._freeze_disabled()

    def assert_zero_residual(self, modalities):
        """Reject a supposedly new branch whose checkpoint already alters RGB."""
        for name in normalize_modalities(modalities):
            for fusion in getattr(self.tri_fusion, name + "_fusions"):
                if any(parameter.detach().count_nonzero().item()
                       for parameter in fusion.output.parameters()):
                    raise RuntimeError("New modality does not have zero residual: " + name)

    def train(self, mode=True):
        super().train(mode)
        if hasattr(self, "_active_modalities"):
            self._freeze_disabled()
        return self

    def extract_feat(self, img, img_metas=None):
        if self.fusion_enabled != bool(self.active_modalities):
            raise RuntimeError("fusion_enabled changed outside the modality route interface")
        if img.ndim != 4 or img.shape[1] != 8:
            raise ValueError("ProgressiveCoDETR requires the eight-channel multimodal input")
        self._freeze_disabled()
        rgb_modules = (self.backbone, getattr(self, 'neck', None))
        frozen_rgb = not any(p.requires_grad for module in rgb_modules if module is not None
                             for p in module.parameters())
        # Rebuild the RGB graph whenever its parameters are trainable.
        with torch.no_grad() if frozen_rgb else nullcontext():
            features = CoDETR.extract_feat(self, img[:, :3].contiguous(), img_metas)
        if not self.active_modalities:
            return features
        fusion = self.tri_fusion
        if not isinstance(features, (list, tuple)) or len(features) != fusion.num_levels:
            raise ValueError("Official SFP must return five feature tensors")
        valid = img[:, 6:7].clamp(0, 1) if "depth" in self.active_modalities else None
        infrared = fusion.ir_pyramid(img[:, 3:4]) if "ir" in self.active_modalities else None
        depth = fusion.depth_pyramid(img[:, 4:8] * valid) if valid is not None else None
        outputs = []
        for index, rgb in enumerate(features):
            if rgb.ndim != 4 or rgb.shape[:2] != (img.shape[0], fusion.fusion_channels):
                raise ValueError("SFP feature batch/channels do not match fusion configuration")
            result = rgb
            if infrared is not None:
                result = result + fusion.ir_fusions[index](rgb, infrared[index])
            if depth is not None:
                result = result + fusion.depth_fusions[index](rgb, depth[index], valid)
            outputs.append(result)
        return tuple(outputs)
