"""Hand-constructed checks for descriptive refiner diagnostics."""
import unittest

import numpy as np

from analysis.refiner_diagnostics import baseline_skeleton_sets, change_counts, fragment_shapes


class RefinerDiagnosticsTests(unittest.TestCase):
    def test_four_flip_categories_partition_valid_flips(self):
        draft = np.array([[0, 0, 1, 1, 0, 1]], dtype=bool)
        prediction = ~draft
        truth = np.array([[1, 0, 0, 1, 1, 0]], dtype=bool)
        valid = np.array([[1, 1, 1, 1, 0, 0]], dtype=bool)
        counts = change_counts(draft, prediction, truth, valid)
        self.assertEqual(counts, dict(added_correct=1, added_wrong=1, deleted_correct=1, deleted_wrong=1))
        self.assertEqual(sum(counts.values()), int(((draft != prediction) & valid).sum()))

    def test_unchanged_predictions_have_no_flips(self):
        mask = np.ones((4, 4), dtype=bool)
        self.assertEqual(sum(change_counts(mask, mask, ~mask, mask).values()), 0)

    def test_single_token_cell_uses_both_inclusive_endpoints(self):
        empty = np.zeros((64, 64), dtype=bool)
        predicted = empty.copy()
        predicted[0:16, 0:16] = True
        predicted[20:22, 30:34] = True  # Small, but straddles columns 31/32.
        shapes = fragment_shapes(predicted, empty, empty, ~empty)
        self.assertEqual(len(shapes), 2)
        square = next(s for s in shapes if s['area'] == 256)
        crossing = next(s for s in shapes if s['area'] == 8)
        self.assertEqual(square['single_cell'], 1)
        self.assertEqual(square['aligned_edges'], 4)
        self.assertEqual(crossing['single_cell'], 0)

    def test_new_fragment_excludes_any_draft_or_truth_overlap(self):
        empty = np.zeros((32, 32), dtype=bool)
        predicted = empty.copy()
        predicted[1:3, 1:3] = True
        predicted[8:10, 8:10] = True
        predicted[20:22, 20:22] = True
        draft, truth = empty.copy(), empty.copy()
        draft[1, 1] = True
        truth[8, 8] = True
        self.assertEqual([s['area'] for s in fragment_shapes(predicted, draft, truth, ~empty)], [4])

    def test_fixed_baseline_sets_match_gap_terminal_and_covered_pixels(self):
        truth = np.zeros((32, 64), dtype=bool)
        truth[16, 2:62] = True
        prediction = truth.copy()
        prediction[16, 2:7] = False
        prediction[16, 25:35] = False
        gaps, terminals, covered = baseline_skeleton_sets(truth, prediction)
        self.assertEqual(int(gaps.sum()), 10)
        self.assertEqual(int(terminals.sum()), 5)
        self.assertEqual(int(covered.sum()), 45)
        self.assertFalse((gaps & terminals).any())


if __name__ == '__main__':
    unittest.main()
