"""CPU-only checks of resize/H direction, pixel alignment and raw support."""

import unittest

import cv2
import numpy as np

from .rectify_qwen import raw_to_source_homography, rectify_raw, resize_matrix


class RectificationTests(unittest.TestCase):
    source_size, normalized_size, raw_size = (64, 48), (80, 60), (106, 88)

    def fitted_h(self, source_to_raw):
        return (resize_matrix(self.normalized_size, (512, 512))
                @ resize_matrix(self.raw_size, self.normalized_size)
                @ source_to_raw @ np.linalg.inv(resize_matrix(self.source_size, (512, 512))))

    def source(self):
        y, x = np.indices((48, 64))
        return np.dstack([3 * x, 4 * y, x + y]).astype(np.uint8)

    def test_pixel_center_resize_coordinates(self):
        transform = resize_matrix((64, 48), (106, 88))
        expected = [(0.5 * 106 / 64 - 0.5), (0.5 * 88 / 48 - 0.5)]
        np.testing.assert_allclose((transform @ [0, 0, 1])[:2], expected)

    def test_affine_conversion_and_exact_pixel_alignment(self):
        source_to_raw = np.array([[1., 0, 21], [0, 1., 19], [0, 0, 1.]])
        converted = raw_to_source_homography(self.fitted_h(source_to_raw), self.source_size, self.normalized_size, self.raw_size)
        np.testing.assert_allclose(converted, np.linalg.inv(source_to_raw), atol=1e-12)
        raw = np.zeros((88, 106, 3), np.uint8)
        raw[19:67, 21:85] = self.source()
        rectified, validity = rectify_raw(raw, converted, self.source_size)
        self.assertTrue(validity.all())
        np.testing.assert_array_equal(rectified[:, :, :3], self.source())

    def test_projective_conversion_and_subpixel_alignment(self):
        source_to_raw = np.array([[1.13, .03, 17], [-.04, 1.11, 14], [.0008, -.0005, 1.]])
        converted = raw_to_source_homography(self.fitted_h(source_to_raw), self.source_size, self.normalized_size, self.raw_size)
        expected = np.linalg.inv(source_to_raw)
        np.testing.assert_allclose(converted, expected / expected[2, 2], atol=1e-12)
        raw = cv2.warpPerspective(self.source(), source_to_raw, self.raw_size, flags=cv2.INTER_LINEAR)
        rectified, validity = rectify_raw(raw, converted, self.source_size)
        self.assertTrue(validity.all())
        difference = np.abs(rectified[4:-4, 4:-4, :3].astype(float) - self.source()[4:-4, 4:-4].astype(float))
        self.assertLess(difference.mean(), 0.6)
        self.assertLessEqual(difference.max(), 2)

    def test_outside_raw_bounds_is_transparent_and_counted(self):
        source_to_raw = np.array([[1., 0, -32], [0, 1., 0], [0, 0, 1.]])
        converted = np.linalg.inv(source_to_raw)
        rectified, validity = rectify_raw(np.ones((48, 64, 3), np.uint8) * 120, converted, (64, 48))
        self.assertEqual(validity.mean(), .5)
        self.assertTrue((rectified[:, :32, 3] == 0).all())
        self.assertTrue((rectified[:, 32:, 3] == 255).all())


if __name__ == "__main__":
    unittest.main()
