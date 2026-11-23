"""Add direct infrared supervision during multimodal feature adaptation."""
_base_ = ['fusion.py']
custom_imports = dict(imports=['trifusion.pipelines', 'trifusion.model',
                              'trifusion.training', 'trifusion.modal'],
                      allow_failed_imports=False)
model = dict(type='ProgressiveCoDETR', active_modalities=('ir', 'depth'))
use_ir_aux = True
