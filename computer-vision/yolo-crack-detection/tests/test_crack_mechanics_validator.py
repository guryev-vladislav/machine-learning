import unittest

import cv2
import numpy as np

from src.utils.crack_mechanics_validator import ConcreteCrackMechanicsValidator


class ConcreteCrackMechanicsValidatorTests(unittest.TestCase):
    def setUp(self):
        self.validator = ConcreteCrackMechanicsValidator()

    def test_evaluate_returns_explainable_metrics(self):
        crop = np.full((64, 64), 150, dtype=np.uint8)
        cv2.line(crop, (12, 32), (52, 32), 30, 2)
        mask = np.zeros((64, 64), dtype=np.uint8)
        cv2.line(mask, (12, 32), (52, 32), 255, 2)

        result = self.validator.evaluate(crop, mask)

        self.assertIn('verified', result)
        self.assertIn('mechanics_confidence', result)
        self.assertEqual(
            set(result['metrics']),
            {'tortuosity', 'texture', 'intensity_profile'},
        )
        self.assertGreaterEqual(result['mechanics_confidence'], 0.0)
        self.assertLessEqual(result['mechanics_confidence'], 1.0)

    def test_empty_mask_is_rejected(self):
        mask = np.zeros((32, 32), dtype=np.uint8)

        result = self.validator.validate_tortuosity(mask)

        self.assertFalse(result['passed'])
        self.assertEqual(result['tortuosity'], 0.0)

    def test_mismatched_crop_and_mask_are_rejected(self):
        crop = np.zeros((32, 32), dtype=np.uint8)
        mask = np.zeros((16, 16), dtype=np.uint8)

        with self.assertRaises(ValueError):
            self.validator.evaluate(crop, mask)


if __name__ == '__main__':
    unittest.main()
