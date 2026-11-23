"""First-stage freezing, native AMP updates and a bounded startup audit."""
import hashlib
import json
from pathlib import Path
from types import MethodType

import torch
from mmcv.runner import HOOKS, Hook, GradientCumulativeFp16OptimizerHook
from mmdet.core.evaluation import EvalHook


def unwrap(model):
    return model.module if hasattr(model, 'module') else model


def freeze_for_stage1(model, phase='fusion'):
    model = unwrap(model)
    if phase not in ('rgb', 'fusion'):
        raise ValueError('Stage1 phase must be rgb or fusion')
    if getattr(model, 'fusion_enabled', True) != (phase == 'fusion'):
        raise RuntimeError('Model fusion_enabled does not match stage1 phase')
    model._stage1_phase = phase
    frozen_prefixes = ('backbone.', 'neck.', 'tri_fusion.') if phase == 'rgb' else ('backbone.', 'neck.')
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(not name.startswith(frozen_prefixes))
    model.backbone.eval()
    model.neck.eval()
    model.tri_fusion.train(phase == 'fusion' and model.training)
    return model


def frozen_digest(model, prefixes=('backbone.', 'neck.')):
    digest = hashlib.sha256()
    for name, value in sorted(unwrap(model).state_dict().items()):
        if name.startswith(prefixes):
            digest.update(name.encode())
            digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def group(name):
    if name.startswith('tri_fusion.ir_'):
        return 'infrared'
    if name.startswith('tri_fusion.depth_'):
        return 'depth'
    if name.startswith('query_head.cls_branches.'):
        return 'classification'
    if name.startswith('query_head.reg_branches.'):
        return 'regression'


def statistics(model):
    model = unwrap(model)
    phase = getattr(model, '_stage1_phase', 'fusion')
    if not hasattr(model, '_stage1_stats'):
        model._stage1_stats = dict(updates=0, amp_skips=0, first80_passed=False,
                                  gradient_groups={}, changed_groups={}, max_shape=[0, 0], phase=phase)
    elif model._stage1_stats.get('phase', 'fusion') != phase:
        raise RuntimeError('Cannot reuse startup audit statistics across phases')
    return model._stage1_stats


def fp32_loss(function):
    def cast(value):
        if isinstance(value, torch.Tensor) and value.is_floating_point():
            return value.float()
        if isinstance(value, (list, tuple)):
            return type(value)(cast(x) for x in value)
        if isinstance(value, dict):
            return {key: cast(x) for key, x in value.items()}
        return value

    def wrapped(*args, **kwargs):
        with torch.cuda.amp.autocast(enabled=False):
            return function(*cast(args), **cast(kwargs))
    return wrapped


@HOOKS.register_module()
class Stage1FreezeHook(Hook):
    def __init__(self, phase='fusion'):
        if phase not in ('rgb', 'fusion'):
            raise ValueError('Stage1 phase must be rgb or fusion')
        self.phase = phase

    def before_run(self, runner):
        model = freeze_for_stage1(runner.model, self.phase)
        if not getattr(model, '_stage1_fp32_loss', False):
            model.query_head.loss = fp32_loss(model.query_head.loss)
            model._stage1_fp32_loss = True

    def before_train_iter(self, runner):
        freeze_for_stage1(runner.model, self.phase)


@HOOKS.register_module()
class Stage1OptimizerHook(GradientCumulativeFp16OptimizerHook):
    def before_run(self, runner):
        super().before_run(runner)
        model = unwrap(runner.model)
        stats = statistics(model)
        original = runner.optimizer.step
        probes = {name: value for name, value in model.named_parameters()
                  if value.requires_grad and (('.output.weight' in name and group(name)) or
                  name in ('query_head.cls_branches.0.weight',
                           'query_head.reg_branches.0.0.weight'))}

        def effective_step(_optimizer, *args, **kwargs):
            # GradScaler invokes this only after unscale/clip and only for a valid step.
            for name, value in model.named_parameters():
                if value.grad is None:
                    continue
                if not value.requires_grad:
                    raise RuntimeError('Frozen parameter received a gradient: ' + name)
                if not torch.isfinite(value.grad).all():
                    raise RuntimeError('Nonfinite effective gradient: ' + name)
                key = group(name)
                if key and torch.count_nonzero(value.grad):
                    stats['gradient_groups'][key] = True
            before = {name: value.detach().clone() for name, value in probes.items()}
            result = original(*args, **kwargs)
            for name, value in probes.items():
                if not torch.equal(value.detach(), before[name]):
                    stats['changed_groups'][group(name)] = True
            stats['updates'] += 1
            return result
        runner.optimizer.step = MethodType(effective_step, runner.optimizer)

    def after_train_iter(self, runner):
        stats = statistics(runner.model)
        before = stats['updates']
        super().after_train_iter(runner)
        if self.every_n_iters(runner, self.cumulative_iters) or self.is_last_iter(runner):
            if stats['updates'] == before:
                stats['amp_skips'] += 1


@HOOKS.register_module()
class Stage1AuditHook(Hook):
    def __init__(self, first_microimages=80, max_memory_fraction=0.9, phase='fusion'):
        if phase not in ('rgb', 'fusion'):
            raise ValueError('Stage1 phase must be rgb or fusion')
        self.phase = phase
        self.first_microimages = first_microimages
        self.max_memory_fraction = max_memory_fraction

    def before_run(self, runner):
        if not torch.cuda.is_available():
            raise RuntimeError('Stage1 training requires an available GPU')
        model = unwrap(runner.model)
        if getattr(model, '_stage1_phase', 'fusion') != self.phase:
            raise RuntimeError('Startup audit phase does not match freeze hook')
        stats = statistics(model)
        resume_stats = runner.meta.get('stage1_audit', {})
        if resume_stats and resume_stats.get('phase', 'fusion') != self.phase:
            raise RuntimeError('Cannot resume startup audit from a different phase')
        stats.update(resume_stats)
        self.initial_digest = frozen_digest(model)
        if stats.get('frozen_digest', self.initial_digest) != self.initial_digest:
            raise RuntimeError('Resume frozen RGB backbone/neck identity changed')
        if self.phase == 'rgb':
            self.initial_fusion_digest = frozen_digest(model, ('tri_fusion.',))
            if stats.get('frozen_fusion_digest', self.initial_fusion_digest) != self.initial_fusion_digest:
                raise RuntimeError('Resume frozen modality branch identity changed')
        self.total_memory = torch.cuda.get_device_properties(0).total_memory
        torch.cuda.reset_peak_memory_stats()
        original = model.forward

        def measured_forward(_model, img, img_metas, *args, **kwargs):
            if kwargs.get('return_loss', True):
                if img.shape[1] != 8 or not torch.isfinite(img).all():
                    raise RuntimeError('Invalid eight-channel training input')
                stats['max_shape'] = [max(a, b) for a, b in zip(stats['max_shape'], img.shape[-2:])]
            return original(img, img_metas, *args, **kwargs)
        model.forward = MethodType(measured_forward, model)

    def after_train_iter(self, runner):
        model = unwrap(runner.model)
        stats = statistics(model)
        loss = runner.outputs['loss']
        if not torch.isfinite(loss):
            raise RuntimeError('Nonfinite training loss')
        peak = torch.cuda.max_memory_reserved()
        if peak >= self.total_memory * self.max_memory_fraction:
            raise RuntimeError('Stage1 memory guard exceeded: %d/%d' % (peak, self.total_memory))
        if not stats['first80_passed'] and runner.iter + 1 >= self.first_microimages:
            required = {'classification', 'regression'}
            if self.phase == 'fusion':
                required.update(('infrared', 'depth'))
            if stats['updates'] < 2 or not required.issubset(stats['gradient_groups']) or not required.issubset(stats['changed_groups']):
                raise RuntimeError('First80 did not demonstrate effective modality/head updates')
            if stats['max_shape'][0] < 864 or stats['max_shape'][1] < 1536:
                raise RuntimeError('First80 has not exercised the maximum configured shape')
            if frozen_digest(model) != self.initial_digest:
                raise RuntimeError('Frozen RGB backbone/neck state changed')
            if self.phase == 'rgb':
                if frozen_digest(model, ('tri_fusion.',)) != self.initial_fusion_digest:
                    raise RuntimeError('Frozen modality branch state changed during RGB adaptation')
                if any(value.grad is not None for value in model.tri_fusion.parameters()):
                    raise RuntimeError('Frozen modality branch received a gradient')
                stats['frozen_fusion_digest'] = self.initial_fusion_digest
            if any(not torch.isfinite(value).all() for value in model.parameters()):
                raise RuntimeError('Nonfinite model parameter')
            stats.update(first80_passed=True, peak_reserved_bytes=peak,
                         gpu_total_bytes=self.total_memory, frozen_digest=self.initial_digest,
                         actual_microimages=runner.iter + 1)
            Path(runner.work_dir, 'FIRST80.json').write_text(json.dumps(stats, indent=2) + '\n')
            runner.logger.info('FIRST80_PASSED updates=%s AMP_skips=%s', stats['updates'], stats['amp_skips'])
        runner.meta['stage1_audit'] = dict(stats)


class Stage1Fp32EvalHook(EvalHook):
    def _do_evaluate(self, runner):
        flags = [(module, module.fp16_enabled) for module in runner.model.modules()
                 if hasattr(module, 'fp16_enabled')]
        try:
            for module, _ in flags:
                module.fp16_enabled = False
            with torch.cuda.amp.autocast(enabled=False):
                return super()._do_evaluate(runner)
        finally:
            for module, value in flags:
                module.fp16_enabled = value
