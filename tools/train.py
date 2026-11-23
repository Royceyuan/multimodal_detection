"""Train or resume a multimodal detector on local COCO-format annotations."""
import argparse
from pathlib import Path
from bootstrap import ROOT, configuration, detector


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, default=ROOT / 'configs/fusion.py')
    p.add_argument('--work-dir', type=Path, required=True)
    p.add_argument('--init-checkpoint', type=Path)
    p.add_argument('--resume', type=Path)
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()
    if args.resume and args.init_checkpoint:
        p.error('Choose either initialization or full-state resume')
    import torch
    from mmdet.apis import set_random_seed
    import importlib
    from mmdet.datasets import build_dataset
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for model training')
    cfg = configuration(args.config)
    cfg.work_dir = str(args.work_dir.resolve())
    cfg.gpu_ids = [0]
    cfg.device = 'cuda'
    cfg.seed = args.seed
    cfg.auto_scale_lr = dict(enable=False)
    cfg.resume_from = str(args.resume.resolve()) if args.resume else None
    if args.resume and not args.resume.is_file():
        raise FileNotFoundError(args.resume)
    set_random_seed(args.seed, deterministic=False)
    model = detector(cfg)
    if args.init_checkpoint:
        saved = torch.load(str(args.init_checkpoint), map_location='cpu')
        state = saved.get('state_dict', saved)
        own = model.state_dict()
        compatible = {k: v for k, v in state.items()
                      if k in own and own[k].shape == v.shape}
        if not compatible:
            raise ValueError('Initialization contains no compatible model tensors')
        model.load_state_dict(compatible, strict=False)
        print(f'Initialized {len(compatible)} compatible tensors; other tensors retain initialization.', flush=True)
    datasets = [build_dataset(cfg.data.train)]
    cfg.checkpoint_config.meta = dict(CLASSES=model.CLASSES)
    from trifusion.training import Stage1Fp32EvalHook
    api = importlib.import_module('mmdet.apis.train')
    api.EvalHook = Stage1Fp32EvalHook
    api.train_detector(model, datasets, cfg, distributed=False, validate=True,
                       meta=dict(CLASSES=model.CLASSES))


if __name__ == '__main__':
    main()
