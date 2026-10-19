"""Small-mask checks for y0 geometry statistics (no weights or dataset reads)."""
import unittest

import numpy as np
from scipy import ndimage as ndi

from analysis.y0_error_analysis import analyze_masks, distance_error_bins


class Y0ErrorAnalysisTests(unittest.TestCase):
    @staticmethod
    def straight_road():
        raw = np.ones((96, 96), dtype=np.uint8)
        raw[48, 8:80] = 3
        return raw

    @staticmethod
    def segment_count(result, kind):
        return sum(item["segments"] for item in result["segments"][kind].values())

    def test_ten_pixel_gap_has_two_contacts(self):
        raw = self.straight_road()
        predicted = raw == 3
        predicted[48, 35:45] = False
        result = analyze_masks(predicted, raw)
        self.assertEqual(self.segment_count(result, "gap"), 1)
        self.assertEqual(result["segments"]["gap"]["9-16"], {"segments": 1, "skeleton_pixels": 10})
        self.assertEqual(result["uncovered_skeleton_pixels"], 10)
        self.assertEqual(self.segment_count(result, "terminal_missing"), 0)
        self.assertEqual(self.segment_count(result, "whole_missed"), 0)

    def test_one_pixel_dilation_only_has_near_boundary_fp(self):
        raw = self.straight_road()
        predicted = ndi.binary_dilation(raw == 3, structure=np.ones((3, 3), bool))
        result = analyze_masks(predicted, raw)
        self.assertGreater(result["fp_pixels"], 0)
        self.assertEqual(result["fp_distance"]["1-2"], result["fp_pixels"])
        self.assertEqual(result["fn_pixels"], 0)

    def test_remote_square_is_one_fragment_with_far_fp(self):
        raw = self.straight_road()
        predicted = raw == 3
        predicted[5:9, 75:79] = True
        result = analyze_masks(predicted, raw)
        self.assertEqual(result["fragment_areas"], [16])
        self.assertEqual(result["fp_distance"][">16"], 16)
        self.assertEqual(result["fp_pixels"], 16)

    def test_empty_prediction_only_has_whole_missed_road(self):
        raw = self.straight_road()
        result = analyze_masks(np.zeros_like(raw, dtype=bool), raw)
        self.assertEqual(self.segment_count(result, "whole_missed"), 1)
        self.assertEqual(self.segment_count(result, "terminal_missing"), 0)
        self.assertEqual(self.segment_count(result, "gap"), 0)
        self.assertEqual(result["fn_distance"][">16"], 72)
        self.assertEqual(result["uncovered_skeleton_pixels"], 72)

    def test_one_contact_is_terminal_missing(self):
        raw = self.straight_road()
        predicted = raw == 3
        predicted[48, 70:80] = False
        result = analyze_masks(predicted, raw)
        self.assertEqual(self.segment_count(result, "terminal_missing"), 1)
        self.assertEqual(result["segments"]["terminal_missing"]["9-16"]["skeleton_pixels"], 10)

    def test_empty_truth_and_ignored_predictions(self):
        raw = np.ones((32, 32), dtype=np.uint8)
        raw[:8] = 0
        predicted = np.zeros_like(raw, dtype=bool)
        predicted[:8] = True
        predicted[20:22, 20:22] = True
        result = analyze_masks(predicted, raw)
        self.assertEqual(result["fp_pixels"], 4)
        self.assertEqual(result["fp_distance"][">16"], 4)
        self.assertEqual(result["fragment_areas"], [4])
        self.assertEqual(result["predicted_components"], 1)
        self.assertEqual(result["skeleton_pixels"], 0)

    def test_continuous_distances_use_upper_bound_bins(self):
        reference = np.zeros((8, 8), dtype=bool)
        reference[0, 0] = True
        errors = np.zeros_like(reference)
        errors[1, 2] = True  # sqrt(5): greater than 2, less than 3.
        self.assertEqual(distance_error_bins(errors, reference)["3-4"], 1)


if __name__ == "__main__":
    unittest.main()
