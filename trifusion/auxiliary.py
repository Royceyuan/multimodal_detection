"""Training-only direct FCOS supervision of the existing five infrared levels."""
from copy import deepcopy
from functools import partial
from types import MethodType

from torch import nn


AUX_WEIGHT = .1
HEAD_CONFIG = dict(
    type='FCOSHead', num_classes=12, in_channels=32, feat_channels=32,
    stacked_convs=2, strides=(4, 8, 16, 32, 64),
    regress_ranges=((-1, 32), (32, 64), (64, 128), (128, 256), (256, 1e8)),
    norm_cfg=dict(type='GN', num_groups=8, requires_grad=True),
    center_sampling=True, norm_on_bbox=True, centerness_on_reg=True)


def make_aux_head():
    from mmdet.models import build_head
    head = build_head(deepcopy(HEAD_CONFIG))
    head.init_weights()
    return head


def _weighted(value):
    if isinstance(value, (tuple, list)):
        return type(value)(_weighted(item) for item in value)
    return value * AUX_WEIGHT


class IRAux(nn.Module):
    """Capture resized IR before projection, gating and addition to RGB."""

    def __init__(self, head=None):
        super().__init__()
        self.head = make_aux_head() if head is None else head
        self._handles = []
        self.clear()

    def bind(self, model):
        self.close()
        fusions = model.tri_fusion.ir_fusions
        if len(fusions) != 5 or any(fusion.output.in_channels != 32 for fusion in fusions):
            raise ValueError('Expected five existing 32-channel infrared fusion inputs')
        self._handles = [fusion.output.register_forward_pre_hook(
            partial(self._capture, index)) for index, fusion in enumerate(fusions)]
        return self

    def begin(self):
        self.clear()
        if not self.training or len(self._handles) != 5:
            raise RuntimeError('IRAux must be bound and training before begin')
        self._features = [None] * 5
        self._active = True

    def _capture(self, index, module, inputs):
        if not (self._active and self.training):
            return
        feature, = inputs
        if feature.ndim != 4 or feature.shape[1] != 32:
            raise ValueError('Expected NCHW resized infrared features with 32 channels')
        if self._features[index] is not None:
            raise RuntimeError('Infrared level captured twice in one training forward')
        self._features[index] = feature

    def loss(self, gt_bboxes, gt_labels, img_metas, gt_bboxes_ignore=None):
        if not (self.training and self._active and len(self._features) == 5
                and all(feature is not None for feature in self._features)):
            raise RuntimeError('Expected all five infrared levels from the current forward')
        losses = self.head.forward_train(tuple(self._features), img_metas,
                                        gt_bboxes, gt_labels,
                                        gt_bboxes_ignore=gt_bboxes_ignore)
        return {'ir_aux_' + key: _weighted(value) for key, value in losses.items()}

    def clear(self):
        self._features = []
        self._active = False

    def close(self):
        self.clear()
        for handle in self._handles:
            handle.remove()
        self._handles.clear()


def attach_auxiliary(model, head=None):
    """Register once inside the detector, so its losses share the DDP reducer."""
    if hasattr(model, 'ir_aux'):
        raise RuntimeError('Infrared auxiliary head already attached')
    original = model.forward_train.__func__
    aux = IRAux(head).bind(model)
    aux.train(model.training)
    model.ir_aux = aux

    def forward_train(core, img, img_metas, gt_bboxes, gt_labels,
                      gt_bboxes_ignore=None, gt_masks=None, proposals=None, **kwargs):
        if not core.training:
            return original(core, img, img_metas, gt_bboxes, gt_labels,
                            gt_bboxes_ignore=gt_bboxes_ignore, gt_masks=gt_masks,
                            proposals=proposals, **kwargs)
        core.ir_aux.begin()
        try:
            losses = original(core, img, img_metas, gt_bboxes, gt_labels,
                              gt_bboxes_ignore=gt_bboxes_ignore, gt_masks=gt_masks,
                              proposals=proposals, **kwargs)
            extra = core.ir_aux.loss(gt_bboxes, gt_labels, img_metas, gt_bboxes_ignore)
            if losses.keys() & extra.keys():
                raise RuntimeError('Infrared auxiliary loss would overwrite a main loss')
            return dict(losses, **extra)
        finally:
            core.ir_aux.clear()

    model.forward_train = MethodType(forward_train, model)
    return aux
