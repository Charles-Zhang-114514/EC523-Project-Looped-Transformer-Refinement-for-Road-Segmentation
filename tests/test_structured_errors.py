"""Analytical checks for shared geometry and local repair denominators."""
import sys
from pathlib import Path
import unittest

import numpy as np
from scipy import ndimage as ndi

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from looproad.eval.structured_errors import (
    CONNECT, component_merges, connected_at, endpoint_connect, false_bridges,
    local_gaps, thin, thicken,
)


class StructuredErrorTests(unittest.TestCase):
    def road(self):
        mask = np.zeros((256, 256), bool)
        mask[123:134, 20:236] = True
        return mask

    def site(self):
        return dict(center=[128, 128], tangent=[0., 1.], width=12.)

    def test_global_cycle_does_not_invalidate_local_gap(self):
        mask = self.road()
        mask[20:129, 25:36] = True
        mask[20:129, 220:231] = True
        mask[20:31, 25:231] = True
        pred, events = local_gaps(mask, [self.site()], 16)
        event = events[0]
        self.assertTrue(connected_at(pred, event['anchors'], 8))
        self.assertEqual(event['status_8'], 'valid')
        self.assertEqual(event['status_4'], 'valid')
        fixed = endpoint_connect(pred, np.ones_like(mask), 32)
        self.assertTrue(connected_at(fixed, event['anchors'], 8, event['window']))

    def test_local_cycle_is_bypassed(self):
        mask = self.road()
        mask[90:129, 90:101] = True
        mask[90:129, 156:167] = True
        mask[90:101, 90:167] = True
        _, events = local_gaps(mask, [self.site()], 16)
        self.assertEqual(events[0]['status_8'], 'bypassed')
        self.assertEqual(events[0]['status_4'], 'bypassed')

    def test_ignore_strip_prevents_endpoint_join(self):
        mask = self.road()
        pred, _ = local_gaps(mask, [self.site()], 16)
        valid = np.ones_like(mask)
        valid[:, 128] = False
        fixed = endpoint_connect(pred, valid, 32)
        self.assertFalse(fixed[:, 128].any())
        self.assertEqual(component_merges(pred, fixed), 0)

    def test_four_and_eight_neighbors_are_distinct(self):
        diagonal = np.eye(5, dtype=bool)
        self.assertTrue(connected_at(diagonal, [[0, 0], [4, 4]], 8))
        self.assertFalse(connected_at(diagonal, [[0, 0], [4, 4]], 4))

    def test_bridge_joins_distinct_components_with_bounded_distance(self):
        mask = np.zeros((256, 256), bool)
        mask[60:71, 30:220] = True
        mask[90:101, 30:220] = True
        pred, events, reason = false_bridges(mask, np.ones_like(mask), np.random.default_rng(523))
        self.assertEqual(reason, 'ok')
        self.assertEqual(len(events), 1)
        self.assertTrue(10 <= events[0]['separation'] <= 60)
        self.assertEqual(ndi.label(mask, CONNECT)[1], 2)
        self.assertEqual(ndi.label(pred, CONNECT)[1], 1)
        self.assertEqual(component_merges(mask, pred), 1)

    def test_bridge_rejects_ignore(self):
        mask = np.zeros((256, 256), bool)
        mask[60:71, 30:220] = True
        mask[90:101, 30:220] = True
        valid = np.ones_like(mask)
        valid[80] = False
        pred, events, reason = false_bridges(mask, valid, np.random.default_rng(523))
        self.assertEqual(events, [])
        self.assertEqual(reason, 'no_eligible_pair')
        np.testing.assert_array_equal(mask, pred)

    def test_distance_offsets_are_nested(self):
        mask = self.road()
        self.assertTrue(np.all(thin(mask, 2) <= thin(mask, 1)))
        self.assertTrue(np.all(thin(mask, 1) <= mask))
        self.assertTrue(np.all(mask <= thicken(mask, 1)))
        self.assertTrue(np.all(thicken(mask, 1) <= thicken(mask, 2)))


if __name__ == '__main__':
    unittest.main()
