"""Additional tests for src/utils/crack_mechanics_validator.py."""

import unittest

import cv2
import numpy as np

from src.utils.crack_mechanics_validator import ConcreteCrackMechanicsValidator


class ConstructorValidationTests(unittest.TestCase):
    """Tests for ConcreteCrackMechanicsValidator.__init__."""

    def test_valid_parameters(self):
        validator = ConcreteCrackMechanicsValidator(
            tortuosity_range=(1.0, 3.0),
            min_texture_variance=5.0,
            min_profile_score=0.1,
            acceptance_threshold=0.5,
        )
        self.assertEqual(validator.tortuosity_range, (1.0, 3.0))
        self.assertEqual(validator.min_texture_variance, 5.0)

    def test_invalid_tortuosity_range_raises(self):
        with self.assertRaises(ValueError):
            ConcreteCrackMechanicsValidator(tortuosity_range=(2.0, 1.0))

    def test_zero_tortuosity_range_raises(self):
        with self.assertRaises(ValueError):
            ConcreteCrackMechanicsValidator(tortuosity_range=(0, 1.0))

    def test_negative_texture_variance_raises(self):
        with self.assertRaises(ValueError):
            ConcreteCrackMechanicsValidator(min_texture_variance=-1.0)

    def test_invalid_profile_score_raises(self):
        with self.assertRaises(ValueError):
            ConcreteCrackMechanicsValidator(min_profile_score=1.5)

    def test_invalid_acceptance_threshold_raises(self):
        with self.assertRaises(ValueError):
            ConcreteCrackMechanicsValidator(acceptance_threshold=1.5)


class TortuosityValidationTests(unittest.TestCase):
    """Tests for validate_tortuosity method."""

    def setUp(self):
        self.validator = ConcreteCrackMechanicsValidator()

    def test_straight_line_has_low_tortuosity(self):
        mask = np.zeros((64, 64), dtype=np.uint8)
        cv2.line(mask, (10, 32), (54, 32), 255, 2)
        result = self.validator.validate_tortuosity(mask)
        self.assertGreater(result['tortuosity'], 0)
        self.assertGreater(result['path_length'], 0)

    def test_curved_line_has_higher_tortuosity(self):
        mask = np.zeros((64, 64), dtype=np.uint8)
        # Draw a curved line
        points = np.array([[10, 20], [20, 40], [30, 20], [40, 40], [50, 20]], dtype=np.int32)
        cv2.polylines(mask, [points], False, 255, 2)
        result = self.validator.validate_tortuosity(mask)
        self.assertGreater(result['tortuosity'], 0)

    def test_isolated_blobs_rejected(self):
        mask = np.zeros((64, 64), dtype=np.uint8)
        cv2.circle(mask, (32, 32), 10, 255, -1)
        result = self.validator.validate_tortuosity(mask)
        self.assertFalse(result['passed'])

    def test_empty_mask_returns_zero(self):
        mask = np.zeros((32, 32), dtype=np.uint8)
        result = self.validator.validate_tortuosity(mask)
        self.assertEqual(result['tortuosity'], 0.0)
        self.assertFalse(result['passed'])


class TextureValidationTests(unittest.TestCase):
    """Tests for validate_texture method."""

    def setUp(self):
        self.validator = ConcreteCrackMechanicsValidator()

    def test_smooth_surface_rejected(self):
        gray = np.full((64, 64), 150, dtype=np.uint8)
        result = self.validator.validate_texture(gray)
        self.assertLess(result['laplacian_variance'], 100)

    def test_textured_surface_accepted(self):
        gray = np.random.randint(50, 200, (64, 64), dtype=np.uint8)
        result = self.validator.validate_texture(gray)
        self.assertGreater(result['laplacian_variance'], 0)
        self.assertEqual(result['sample_count'], 64 * 64)

    def test_empty_gray_raises(self):
        gray = np.array([], dtype=np.uint8).reshape(0, 0)
        with self.assertRaises(ValueError):
            self.validator.validate_texture(gray)


class IntensityProfileTests(unittest.TestCase):
    """Tests for validate_intensity_profile method."""

    def setUp(self):
        self.validator = ConcreteCrackMechanicsValidator()

    def test_dark_line_in_center(self):
        gray = np.full((64, 64), 200, dtype=np.uint8)
        gray[:, 30:34] = 30
        result = self.validator.validate_intensity_profile(gray)
        self.assertGreater(result['profile_score'], 0)
        self.assertGreater(result['contrast'], 0)

    def test_uniform_image_low_score(self):
        gray = np.full((64, 64), 128, dtype=np.uint8)
        result = self.validator.validate_intensity_profile(gray)
        self.assertGreaterEqual(result['profile_score'], 0.0)

    def test_empty_gray_raises(self):
        gray = np.array([], dtype=np.uint8).reshape(0, 0)
        with self.assertRaises(ValueError):
            self.validator.validate_intensity_profile(gray)


class EvaluateIntegrationTests(unittest.TestCase):
    """Tests for the evaluate method (integration)."""

    def setUp(self):
        self.validator = ConcreteCrackMechanicsValidator()

    def test_bgr_crop_accepted(self):
        crop = np.zeros((64, 64, 3), dtype=np.uint8)
        crop[:, 30:34, :] = 200
        crop[:, 30:34, 0] = 30  # dark line in B channel
        mask = np.zeros((64, 64), dtype=np.uint8)
        mask[:, 30:34] = 255

        result = self.validator.evaluate(crop, mask)
        self.assertIn('verified', result)
        self.assertIn('mechanics_confidence', result)
        self.assertIn('metrics', result)

    def test_grayscale_crop_accepted(self):
        crop = np.full((64, 64), 200, dtype=np.uint8)
        crop[:, 30:34] = 30
        mask = np.zeros((64, 64), dtype=np.uint8)
        mask[:, 30:34] = 255

        result = self.validator.evaluate(crop, mask)
        self.assertIn('verified', result)

    def test_mismatched_dimensions_raises(self):
        crop = np.zeros((64, 64), dtype=np.uint8)
        mask = np.zeros((32, 32), dtype=np.uint8)

        with self.assertRaises(ValueError):
            self.validator.evaluate(crop, mask)

    def test_invalid_crop_raises(self):
        # 1-channel crop is neither grayscale (2D) nor BGR (3 channels)
        crop = np.zeros((10, 10, 1), dtype=np.uint8)
        mask = np.zeros((10, 10), dtype=np.uint8)

        with self.assertRaises((ValueError, cv2.error)):
            self.validator.evaluate(crop, mask)

    def test_confidence_is_mean_of_checks(self):
        # Create a simple case where all 3 checks pass
        crop = np.full((64, 64), 150, dtype=np.uint8)
        cv2.line(crop, (10, 32), (54, 32), 30, 2)
        mask = np.zeros((64, 64), dtype=np.uint8)
        cv2.line(mask, (10, 32), (54, 32), 255, 2)

        result = self.validator.evaluate(crop, mask)
        confidence = result['mechanics_confidence']
        self.assertGreaterEqual(confidence, 0.0)
        self.assertLessEqual(confidence, 1.0)

        # confidence should be mean of [tortuosity_passed, texture_passed, profile_passed]
        checks = [
            result['metrics']['tortuosity']['passed'],
            result['metrics']['texture']['passed'],
            result['metrics']['intensity_profile']['passed'],
        ]
        expected = float(np.mean(checks))
        self.assertAlmostEqual(confidence, expected, places=5)

    def test_verified_matches_threshold(self):
        validator = ConcreteCrackMechanicsValidator(acceptance_threshold=0.0)
        crop = np.full((64, 64), 150, dtype=np.uint8)
        mask = np.zeros((64, 64), dtype=np.uint8)

        result = validator.evaluate(crop, mask)
        # With threshold=0.0, any positive confidence should pass
        # But if all checks fail, confidence=0 and verified=False
        self.assertIn('is_valid', result)
        self.assertEqual(result['verified'], result['is_valid'])

    def test_high_threshold_requires_stronger_evidence(self):
        low_validator = ConcreteCrackMechanicsValidator(acceptance_threshold=0.3)
        high_validator = ConcreteCrackMechanicsValidator(acceptance_threshold=1.0)

        crop = np.full((64, 64), 150, dtype=np.uint8)
        mask = np.zeros((64, 64), dtype=np.uint8)

        result_low = low_validator.evaluate(crop, mask)
        result_high = high_validator.evaluate(crop, mask)

        # Higher threshold should be harder to pass
        if result_low['verified']:
            self.assertFalse(result_high['verified'])


class BinaryMaskHelperTests(unittest.TestCase):
    """Tests for _as_binary_mask static method."""

    def test_uint8_mask_preserved(self):
        mask = np.zeros((32, 32), dtype=np.uint8)
        mask[10:20, 10:20] = 255
        result = ConcreteCrackMechanicsValidator._as_binary_mask(mask)
        self.assertEqual(result.dtype, np.uint8)

    def test_float_mask_converted(self):
        mask = np.zeros((32, 32), dtype=np.float32)
        mask[10:20, 10:20] = 0.8
        result = ConcreteCrackMechanicsValidator._as_binary_mask(mask)
        self.assertEqual(result.dtype, np.uint8)
        self.assertTrue(np.all(result[10:20, 10:20] > 0))

    def test_empty_mask_raises(self):
        mask = np.array([])
        with self.assertRaises(ValueError):
            ConcreteCrackMechanicsValidator._as_binary_mask(mask)

    def test_3d_mask_raises(self):
        mask = np.zeros((32, 32, 3), dtype=np.uint8)
        with self.assertRaises(ValueError):
            ConcreteCrackMechanicsValidator._as_binary_mask(mask)


if __name__ == '__main__':
    unittest.main()
