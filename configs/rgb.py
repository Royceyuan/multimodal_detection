"""Adapt detection heads while retaining the RGB representation."""
_base_ = ['fusion.py']
stage_phase = 'rgb'
model = dict(fusion_enabled=False)
custom_hooks = [dict(type='Stage1FreezeHook', phase='rgb', priority='VERY_HIGH')]
