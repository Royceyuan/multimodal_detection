"""Resolve the actual MMCV configuration inheritance without loading CUDA ops."""
from pathlib import Path
import importlib.util
import unittest


@unittest.skipUnless(importlib.util.find_spec('mmcv'), 'Install the configured MMCV dependency')
class ConfigurationTests(unittest.TestCase):
    def test_model_and_optimizer_configs(self):
        from mmcv import Config
        root = Path(__file__).resolve().parents[1]
        for name in ('rgb', 'fusion', 'infrared_aux'):
            with self.subTest(config=name):
                cfg = Config.fromfile(str(root/'configs'/f'{name}.py'), import_custom_modules=False)
                self.assertEqual(cfg.model.query_head.num_classes, len(cfg.classes))
                self.assertEqual(cfg.model.roi_head[0].bbox_head.num_classes, len(cfg.classes))
                self.assertEqual(cfg.model.bbox_head[0].num_classes, len(cfg.classes))
                self.assertEqual(cfg.optimizer_config.type, 'Stage1OptimizerHook')
                self.assertIsNone(cfg.fp16)
                if name == 'infrared_aux':
                    self.assertTrue(cfg.use_ir_aux)
                    self.assertEqual(cfg.model.type, 'ProgressiveCoDETR')


if __name__ == '__main__':
    unittest.main()
