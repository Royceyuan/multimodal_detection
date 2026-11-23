"""Resolve the project and bundled framework without machine-specific paths."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'vendor/Co-DETR')]


def configuration(path):
    from mmcv import Config
    from mmcv.utils import import_modules_from_strings
    cfg = Config.fromfile(str(path))
    import_modules_from_strings(**cfg.custom_imports)
    cfg.model.pretrained = None
    cfg.model.backbone.pretrained = None
    cfg.model.backbone.init_cfg = None
    return cfg


def detector(cfg):
    from mmdet.models import build_detector
    model = build_detector(cfg.model, train_cfg=cfg.get('train_cfg'),
                           test_cfg=cfg.get('test_cfg'))
    model.init_weights()
    if cfg.get('use_ir_aux', False):
        from trifusion.auxiliary import attach_auxiliary
        attach_auxiliary(model)
    model.CLASSES = tuple(cfg.classes)
    return model
