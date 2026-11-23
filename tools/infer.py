"""Run multi-scale, flip and optional tiled inference on aligned local images."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
from bootstrap import ROOT, configuration, detector
from trifusion.boxes import Row
from trifusion.fusion import weighted_fusion, merge_tiles, tile_windows
from trifusion.views import load_eight, one_file


def predict(model, cfg, data_root, path, scale, flip=False, window=None):
    import numpy as np
    import torch
    from mmcv.parallel import collate, scatter
    from mmdet.datasets import replace_ImageToTensor
    from mmdet.datasets.pipelines import Compose
    image = load_eight(path, data_root/'infrared', data_root/'depth', window, flip)
    h, w = image.shape[:2]
    pipeline = deepcopy(cfg.data.test.pipeline[1:])
    for stage in pipeline:
        if stage['type'] == 'MultiScaleFlipAug':
            stage['img_scale'] = tuple(scale)
            stage['flip'] = False
            for step in stage['transforms']:
                if step['type'] == 'ResizeAlignedMultimodal':
                    step['img_scale'] = tuple(scale)
    prepared = Compose(replace_ImageToTensor(pipeline))(
        dict(img=image, filename=str(path), ori_filename=path.name,
             img_shape=image.shape, ori_shape=image.shape, img_fields=['img']))
    data = collate([prepared], samples_per_gpu=1)
    data['img_metas'] = [item.data[0] for item in data['img_metas']]
    data['img'] = [item.data[0] for item in data['img']]
    data = scatter(data, [next(model.parameters()).device])[0]
    with torch.no_grad():
        predictions = model(return_loss=False, rescale=True, **data)[0]
    if isinstance(predictions, tuple):
        predictions = predictions[0]
    rows = []
    for cls, boxes in enumerate(predictions):
        for x1, y1, x2, y2, score in np.asarray(boxes).reshape(-1, 5):
            if not np.isfinite([x1, y1, x2, y2, score]).all() or not 0 <= score <= 1:
                raise ValueError('Nonfinite or invalid model prediction')
            if flip:
                x1, x2 = w-x2, w-x1
            x1, x2 = np.clip([x1, x2], 0, w)
            y1, y2 = np.clip([y1, y2], 0, h)
            if x2 > x1 and y2 > y1:
                rows.append(Row(cls, float((x1+x2)/(2*w)), float((y1+y2)/(2*h)),
                                float((x2-x1)/w), float((y2-y1)/h), float(score)))
    return sorted(rows, key=lambda row: -row.confidence)[:300]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, default=ROOT/'configs/fusion.py')
    for name in ('data', 'checkpoint', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--scales', nargs='+', type=int, default=[800, 960, 1067, 1280])
    p.add_argument('--flip', action='store_true')
    p.add_argument('--tiles', action='store_true')
    p.add_argument('--tile-size', nargs=2, type=int, default=[1100, 620])
    p.add_argument('--score-thr', type=float, default=0.3)
    p.add_argument('--max-detections', type=int, default=100)
    args = p.parse_args()
    if (any(s <= 0 for s in args.scales) or not 0 <= args.score_thr <= 1
            or min(args.tile_size) < 1 or args.max_detections < 1):
        p.error('Invalid scale, tile size, confidence or detection limit')
    if args.output.exists():
        raise FileExistsError('Use a fresh output directory')
    from PIL import Image
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for detector inference')
    cfg = configuration(args.config)
    model = detector(cfg)
    model.query_head.test_cfg['max_per_img'] = 300
    saved = torch.load(str(args.checkpoint), map_location='cpu')
    model.load_state_dict(saved.get('state_dict', saved), strict=True)
    del saved
    for module in model.modules():
        if hasattr(module, 'fp16_enabled'):
            module.fp16_enabled = False
    model.requires_grad_(False)
    model = model.float().cuda().eval()
    rgb_root = args.data/'RGB'
    paths = sorted(path for path in rgb_root.iterdir() if path.suffix.lower() in ('.png', '.jpg', '.jpeg', '.bmp'))
    if not paths or len({path.stem for path in paths}) != len(paths):
        raise ValueError('Require nonempty RGB inputs with unique file stems')
    for path in paths:
        for modality in ('infrared', 'depth'):
            one_file(args.data/modality, path.stem)
    args.output.mkdir(parents=True)
    for index, path in enumerate(paths, 1):
        with Image.open(path) as image:
            width, height = image.size
        views = []
        for size in args.scales:
            scale = (round(size*16/9), size)
            for flip in ([False, True] if args.flip else [False]):
                views.append(predict(model, cfg, args.data, path, scale, flip))
        base = weighted_fusion(views)
        base = [row for row in base if row.confidence >= args.score_thr]
        tiles = []
        if args.tiles:
            for window in tile_windows(width, height, tuple(args.tile_size)):
                x1, y1, x2, y2 = window
                for row in predict(model, cfg, args.data, path, (1707, 960), window=window):
                    if row.confidence >= args.score_thr:
                        tiles.append(Row(row.class_id, (row.x*(x2-x1)+x1)/width,
                                         (row.y*(y2-y1)+y1)/height,
                                         row.w*(x2-x1)/width, row.h*(y2-y1)/height, row.confidence))
        final = merge_tiles(base, tiles, limit=args.max_detections)
        content = ''.join(f'{r.class_id} {r.x:.8f} {r.y:.8f} {r.w:.8f} {r.h:.8f} {r.confidence:.8f}\n' for r in final)
        (args.output/(path.stem+'.txt')).write_text(content, encoding='utf-8')
        print(json.dumps(dict(completed=index, total=len(paths), image=path.name, detections=len(final))), flush=True)


if __name__ == '__main__':
    main()
