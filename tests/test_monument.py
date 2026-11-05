import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import monuments


class MonumentTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.db = self.root / 'catalog.sqlite3'
        self.arch = self.root / 'arch.png'
        self.tower = self.root / 'tower.png'
        img = Image.new('L', (48, 48), 0)
        draw = ImageDraw.Draw(img)
        draw.rectangle((7, 7, 41, 42), fill=255)
        draw.rectangle((17, 22, 31, 42), fill=0)
        img.save(self.arch)
        img = Image.new('L', (48, 48), 0)
        ImageDraw.Draw(img).rectangle((21, 3, 27, 43), fill=255)
        img.save(self.tower)

    def tearDown(self):
        self.temp.cleanup()

    def test_feature_shape_and_determinism(self):
        first = monuments.extract_features(self.arch)
        self.assertEqual(first.shape, (364,))
        np.testing.assert_array_equal(first, monuments.extract_features(self.arch))

    def test_nearest_reference_and_description(self):
        monuments.add_reference(self.db, self.tower, 'Tower', 'Test tower')
        monuments.add_reference(self.db, self.arch, 'Arch', 'Test arch')
        result = monuments.match(self.db, self.arch)
        self.assertEqual(result['label'], 'Arch')
        self.assertEqual(result['details'], 'Test arch')
        self.assertEqual(result['distance'], 0.0)
        out = self.root / 'comparison.png'
        monuments.save_comparison(self.arch, result, out)
        with Image.open(out) as im:
            self.assertEqual(im.size, (660, 300))

    def test_missing_or_empty_catalog(self):
        with self.assertRaisesRegex(ValueError, 'No catalogue'):
            monuments.match(self.db, self.arch)
        monuments.open_catalog(self.db).close()
        with self.assertRaisesRegex(ValueError, 'No reference images'):
            monuments.match(self.db, self.arch)

    def test_invalid_image_and_blank_label(self):
        bad = self.root / 'bad.png'
        bad.write_text('not an image')
        with self.assertRaises(ValueError):
            monuments.add_reference(self.db, bad, 'Bad')
        with self.assertRaises(ValueError):
            monuments.add_reference(self.db, self.arch, '  ')

    def test_cli_rejects_unknown_catalog(self):
        self.assertEqual(monuments.main(['--db', str(self.db), 'match', str(self.arch)]), 1)


if __name__ == '__main__':
    unittest.main()
