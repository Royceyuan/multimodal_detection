"""MMCV 1.7 deformable-attention FP32 CUDA compatibility patch."""
import argparse
import ast
import hashlib
import importlib.util
from pathlib import Path


def patch_text(text):
    if 'ctx.value_dtype = value.dtype' in text:
        assert 'grad_output = grad_output.float()' in text and 'return output.to(ctx.value_dtype)' in text
        return text
    forward_call_old = '        ctx.im2col_step = im2col_step\n        output = ext_module.ms_deform_attn_forward(\n'
    forward_call_new = '        ctx.im2col_step = im2col_step\n        ctx.value_dtype = value.dtype\n        ctx.sampling_locations_dtype = sampling_locations.dtype\n        ctx.attention_weights_dtype = attention_weights.dtype\n        value = value.float()\n        sampling_locations = sampling_locations.float()\n        attention_weights = attention_weights.float()\n        output = ext_module.ms_deform_attn_forward(\n'
    if text.count(forward_call_old) != 1:
        raise RuntimeError('Expected one MMCV deformable-attention forward call')
    text = text.replace(forward_call_old, forward_call_new, 1)
    forward_return_old = '        ctx.save_for_backward(value, value_spatial_shapes,\n                              value_level_start_index, sampling_locations,\n                              attention_weights)\n        return output'
    forward_return_new = '        ctx.save_for_backward(value, value_spatial_shapes,\n                              value_level_start_index, sampling_locations,\n                              attention_weights)\n        return output.to(ctx.value_dtype)'
    if text.count(forward_return_old) != 1:
        raise RuntimeError('Expected one MMCV deformable-attention forward return')
    text = text.replace(forward_return_old, forward_return_new, 1)
    backward_start = text.find('        value, value_spatial_shapes, value_level_start_index')
    backward_body_end = text.find('        grad_value = torch.zeros_like(value)', backward_start)
    backward_body_old = text[backward_start:backward_body_end]
    if backward_start < 0 or backward_body_end < 0 or 'sampling_locations, attention_weights = ctx.saved_tensors' not in backward_body_old:
        raise RuntimeError('Expected one MMCV deformable-attention backward input block')
    backward_body_new = '        value, value_spatial_shapes, value_level_start_index, sampling_locations, attention_weights = (\n            ctx.saved_tensors)\n        grad_output = grad_output.float()\n'
    text = text[:backward_start] + backward_body_new + text[backward_body_end:]
    backward_return_start = text.find('        return grad_value', backward_body_end)
    backward_return_end = text.find('\n', text.find('grad_attn_weight, None', backward_return_start)) + 1
    if backward_return_start < 0 or backward_return_end <= backward_return_start:
        raise RuntimeError('Expected one MMCV deformable-attention backward return')
    backward_return_new = '        return (grad_value.to(ctx.value_dtype), None, None,\n                grad_sampling_loc.to(ctx.sampling_locations_dtype),\n                grad_attn_weight.to(ctx.attention_weights_dtype), None)\n'
    return text[:backward_return_start] + backward_return_new + text[backward_return_end:]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, help='覆盖自动发现路径；仅用于CPU补丁复现检查')
    p.add_argument('--check-only', action='store_true', help='只验证补丁，不写入')
    args=p.parse_args()
    if args.source:
        path=args.source
    else:
        spec=importlib.util.find_spec('mmcv')
        if spec is None:
            raise RuntimeError('请先在独立虚拟环境安装mmcv-full==1.7.0')
        path=Path(spec.origin).parent/'ops/multi_scale_deform_attn.py'
    before=path.read_text(encoding='utf-8')
    after=patch_text(before)
    ast.parse(after)
    if not args.check_only and after!=before:
        backup=path.with_suffix('.py.fp32_original')
        if backup.exists() and backup.read_text(encoding='utf-8')!=before:
            raise RuntimeError('原始备份冲突，拒绝覆盖')
        backup.write_text(before,encoding='utf-8')
        path.write_text(after,encoding='utf-8')
    print('MMCV_MSDA_FP32_PATCH', 'CHECK_ONLY' if args.check_only else 'APPLIED',hashlib.sha256(after.encode()).hexdigest())


if __name__=='__main__':
    main()
