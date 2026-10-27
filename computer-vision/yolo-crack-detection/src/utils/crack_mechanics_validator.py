"""Deterministic physical plausibility checks for candidate concrete cracks."""

import logging
from typing import Any, Dict, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

try:
    from src.utils.logging_setup import get_logger
except ImportError as exc:
    logger.error('Failed to import project logging setup: %s', exc)
    raise
else:
    logger = get_logger(__name__)


class ConcreteCrackMechanicsValidator:
    """Validate whether an image candidate has geometry and texture of a crack.

    The checks are deliberately conservative heuristics. They do not prove that
    a candidate is concrete; they provide a deterministic second-stage score
    that can reject geometrically or photometrically implausible detections.
    """

    def __init__(
        self,
        tortuosity_range: Tuple[float, float] = (1.05, 2.5),
        min_texture_variance: float = 10.0,
        min_profile_score: float = 0.15,
        acceptance_threshold: float = 0.6,
    ):
        if tortuosity_range[0] <= 0 or tortuosity_range[1] < tortuosity_range[0]:
            raise ValueError('tortuosity_range must be a positive ascending range')
        if min_texture_variance < 0:
            raise ValueError('min_texture_variance must be non-negative')
        if not 0 <= min_profile_score <= 1:
            raise ValueError('min_profile_score must be between 0 and 1')
        if not 0 <= acceptance_threshold <= 1:
            raise ValueError('acceptance_threshold must be between 0 and 1')

        self.tortuosity_range = tortuosity_range
        self.min_texture_variance = float(min_texture_variance)
        self.min_profile_score = float(min_profile_score)
        self.acceptance_threshold = float(acceptance_threshold)

    @staticmethod
    def _as_binary_mask(roi_mask: np.ndarray) -> np.ndarray:
        mask = np.asarray(roi_mask)
        if mask.ndim != 2 or mask.size == 0:
            raise ValueError('roi_mask must be a non-empty two-dimensional array')
        if mask.dtype != np.uint8:
            mask = mask.astype(np.float32)
            mask = (mask > 0).astype(np.uint8) * 255
        else:
            mask = (mask > 0).astype(np.uint8) * 255
        return mask

    @staticmethod
    def _skeletonize(mask: np.ndarray) -> np.ndarray:
        """Return a one-pixel skeleton using morphological erosion/opening."""
        skeleton = np.zeros_like(mask)
        work = mask.copy()
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        while cv2.countNonZero(work):
            eroded = cv2.erode(work, kernel)
            opened = cv2.morphologyEx(eroded, cv2.MORPH_OPEN, kernel)
            skeleton = cv2.bitwise_or(skeleton, cv2.subtract(eroded, opened))
            work = eroded
        return skeleton

    def validate_tortuosity(self, roi_mask: np.ndarray) -> Dict[str, Any]:
        """Measure crack path length divided by its endpoint distance.

        A concrete crack normally has a connected, slightly meandering path.
        The returned ``tortuosity`` is therefore expected in
        ``tortuosity_range``; isolated blobs and nearly straight marks fail.
        """
        mask = self._as_binary_mask(roi_mask)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not contours:
            return {
                'tortuosity': 0.0,
                'path_length': 0.0,
                'euclidean_distance': 0.0,
                'passed': False,
            }

        contour = max(contours, key=cv2.contourArea)
        skeleton = self._skeletonize(mask)
        points = np.column_stack(np.where(skeleton > 0))
        if len(points) < 2:
            return {
                'tortuosity': 0.0,
                'path_length': 0.0,
                'euclidean_distance': 0.0,
                'passed': False,
            }

        # Ordering by principal-axis projection gives a stable approximation
        # of the path for the thin crack masks produced by segmentation.
        centered = points - points.mean(axis=0)
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
        order = np.argsort(centered @ vh[0])
        ordered = points[order].astype(np.float32)
        path_length = float(np.linalg.norm(np.diff(ordered, axis=0), axis=1).sum())
        euclidean_distance = float(np.linalg.norm(ordered[-1] - ordered[0]))
        tortuosity = path_length / euclidean_distance if euclidean_distance > 1e-6 else 0.0
        lower, upper = self.tortuosity_range
        passed = lower <= tortuosity <= upper

        return {
            'tortuosity': tortuosity,
            'path_length': path_length,
            'euclidean_distance': euclidean_distance,
            'contour_area': float(cv2.contourArea(contour)),
            'passed': passed,
        }

    def validate_texture(self, roi_gray: np.ndarray) -> Dict[str, Any]:
        """Estimate concrete roughness from peripheral Laplacian variance.

        The center of a crack can be dark and smooth, so the Laplacian is
        measured on a one-pixel morphological ring around the candidate.
        Higher variance indicates aggregate-scale texture rather than a smooth
        plastic, metal, or gel surface.
        """
        gray = np.asarray(roi_gray)
        if gray.ndim != 2 or gray.size == 0:
            raise ValueError('roi_gray must be a non-empty two-dimensional array')
        gray = cv2.normalize(gray.astype(np.uint8), None, 0, 255, cv2.NORM_MINMAX)
        edges = cv2.Canny(gray, 50, 150)
        ring = cv2.dilate(edges, np.ones((5, 5), np.uint8), iterations=1)
        values = cv2.Laplacian(gray, cv2.CV_64F)[ring > 0]
        variance = float(np.var(values)) if values.size else 0.0
        return {
            'laplacian_variance': variance,
            'sample_count': int(values.size),
            'passed': variance >= self.min_texture_variance,
        }

    def validate_intensity_profile(self, roi_gray: np.ndarray) -> Dict[str, Any]:
        """Score a dark V/U-shaped cross-section characteristic of a crack.

        The profile is sampled through the darkest local region. A valid
        profile has a center darker than both sides and a positive recovery
        toward the outer samples, which approximates shadowing inside a crack.
        """
        gray = np.asarray(roi_gray)
        if gray.ndim != 2 or gray.size == 0:
            raise ValueError('roi_gray must be a non-empty two-dimensional array')
        gray = gray.astype(np.float32)
        blurred = cv2.GaussianBlur(gray, (0, 0), 1.0)
        row, col = np.unravel_index(np.argmin(blurred), blurred.shape)
        half_width = max(2, min(gray.shape) // 8)
        x0 = max(0, col - half_width)
        x1 = min(gray.shape[1], col + half_width + 1)
        profile = blurred[row, x0:x1]
        center = float(profile[len(profile) // 2])
        side_width = max(1, len(profile) // 4)
        side_mean = float(np.mean(np.r_[profile[:side_width], profile[-side_width:]]))
        contrast = max(0.0, side_mean - center)
        scale = max(float(np.std(profile)), 1.0)
        score = float(np.clip(contrast / (3.0 * scale), 0.0, 1.0))
        return {
            'profile_score': score,
            'center_intensity': center,
            'side_intensity': side_mean,
            'contrast': contrast,
            'passed': score >= self.min_profile_score,
        }

    def evaluate(self, roi_crop: np.ndarray, roi_mask: np.ndarray) -> Dict[str, Any]:
        """Combine geometry, substrate texture, and cross-section evidence.

        The score is the mean of the three binary physical checks. The result
        contains both ``verified`` and ``mechanics_confidence`` keys so callers
        can use a clear verdict while retaining an explainable score.
        """
        crop = np.asarray(roi_crop)
        if crop.ndim == 3:
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        elif crop.ndim == 2:
            gray = crop
        else:
            raise ValueError('roi_crop must be a grayscale or BGR image')
        mask = self._as_binary_mask(roi_mask)
        if mask.shape != gray.shape:
            raise ValueError('roi_crop and roi_mask must have identical dimensions')

        tortuosity = self.validate_tortuosity(mask)
        texture = self.validate_texture(gray)
        profile = self.validate_intensity_profile(gray)
        checks = [tortuosity['passed'], texture['passed'], profile['passed']]
        confidence = float(np.mean(checks))
        verified = confidence >= self.acceptance_threshold
        return {
            'verified': verified,
            'is_valid': verified,
            'mechanics_confidence': confidence,
            'confidence': confidence,
            'metrics': {
                'tortuosity': tortuosity,
                'texture': texture,
                'intensity_profile': profile,
            },
        }
