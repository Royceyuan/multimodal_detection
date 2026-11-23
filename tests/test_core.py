"""Synthetic checks for aligned inputs and geometry-based result fusion."""
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
import numpy as np
from PIL import Image
from trifusion.boxes import Row
from trifusion.fusion import weighted_fusion, merge_tiles, tile_windows
from trifusion.views import load_eight
from trifusion.pipelines import build_eight_channel, resize_eight_channel


class CoreTests(unittest.TestCase):
    def test_modal_encoding_and_transforms(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for name in ('RGB', 'infrared', 'depth'):
                (root/name).mkdir()
            rgb = np.arange(12*16*3, dtype=np.uint8).reshape(12, 16, 3)
            depth = np.full((12, 16), 1000, dtype=np.uint16)
            depth[0, 0] = 0
            Image.fromarray(rgb).save(root/'RGB/sample.png')
            Image.fromarray(np.full((12, 16), 128, dtype=np.uint8)).save(root/'infrared/sample.png')
            Image.fromarray(depth).save(root/'depth/sample.png')
            full = load_eight(root/'RGB/sample.png', root/'infrared', root/'depth')
            other = build_eight_channel(root/'RGB/sample.png', root/'infrared', root/'depth')
            self.assertTrue(np.array_equal(full, other))
            self.assertEqual(full.shape, (12, 16, 8))
            self.assertTrue(np.isfinite(full).all())
            self.assertEqual(full[0, 0, 6], 0)
            self.assertEqual(full[1, 1, 6], 1)
            self.assertTrue(np.all(full[:, :, 7] == 1))
            crop = load_eight(root/'RGB/sample.png', root/'infrared', root/'depth', [2, 3, 12, 10], True)
            self.assertTrue(np.array_equal(crop, full[3:10, 2:12][:, ::-1]))
            resized = resize_eight_channel(full, (32, 24))
            self.assertLessEqual(set(np.unique(resized[:, :, 6])), {0, 1})
            with self.assertRaises(ValueError):
                load_eight(root/'RGB/sample.png', root/'infrared', root/'depth', [-1, 0, 4, 4])

    def test_weighted_fusion_preserves_classes_and_votes(self):
        a = Row(0, .5, .5, .2, .2, .9)
        b = replace(a, x=.51, confidence=.8)
        result = weighted_fusion([[a], [b]], [1, 2])
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].confidence, .9)
        expected = (.5*.9**3 + .51*.8**3*2) / (.9**3 + .8**3*2)
        self.assertAlmostEqual(result[0].x, expected)
        self.assertEqual(len(weighted_fusion([[a], [replace(b, class_id=1)]])), 2)
        self.assertEqual(len(weighted_fusion([[a, a], [b]])), 1)
        self.assertEqual(len(weighted_fusion([[a], [replace(b, x=.9)]])), 2)
        with self.assertRaises(ValueError):
            weighted_fusion([[a]], [float('nan')])

    def test_tile_containment_and_class_isolation(self):
        base = Row(0, .5, .5, .02, .02, .9)
        contained = replace(base, h=.008, confidence=.8)
        self.assertEqual(len(merge_tiles([base], [contained])), 1)
        self.assertEqual(len(merge_tiles([base], [replace(contained, class_id=1)])), 2)
        self.assertEqual(len(merge_tiles([base], [replace(contained, x=.8)])), 2)
        self.assertEqual(len(merge_tiles([base], [replace(contained, x=.8)], limit=1)), 1)
        with self.assertRaises(ValueError):
            merge_tiles([base], [], ios=0)

    def test_tiles_cover_arbitrary_image_edges(self):
        windows = tile_windows(1920, 1080)
        covered = np.zeros((1080, 1920), dtype=bool)
        for x1, y1, x2, y2 in windows:
            self.assertTrue(0 <= x1 < x2 <= 1920 and 0 <= y1 < y2 <= 1080)
            covered[y1:y2, x1:x2] = True
        self.assertTrue(covered.all())
        self.assertEqual(tile_windows(320, 240), [(0, 0, 320, 240)])
        with self.assertRaises(ValueError):
            tile_windows(320, 240, overlap=1)


if __name__ == '__main__':
    unittest.main()
