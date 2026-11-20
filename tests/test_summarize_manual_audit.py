"""Check all 30 audit review numbers join to their keyed cases."""
from pathlib import Path
import tempfile
import unittest
from analysis.summarize_manual_audit import joined_rows


class ManualAuditTests(unittest.TestCase):
    def test_all_thirty_reviews_join_without_missing_numbers(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as tmp:
            key = Path(tmp) / "key.csv"
            key.write_text("audit_no,id,delta_A4_y0,added_pixels,added_precision\n" +
                           "".join(f"{i},urban/{i},-0.01,100,0.2\n" for i in range(1, 31)))
            rows = joined_rows(root / "results/manual_audit/audit_sheet_dingning.csv", key)
        self.assertEqual(len(rows), 30)
        self.assertEqual({int(row["audit_no"]) for row in rows}, set(range(1, 31)))
        self.assertTrue(all(row["id"] and row["delta_A4_y0"] and row["added_precision"] for row in rows))

    def test_missing_key_is_rejected(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as tmp:
            key = Path(tmp) / "key.csv"
            key.write_text("audit_no,id,delta_A4_y0,added_pixels,added_precision\n")
            with self.assertRaisesRegex(ValueError, "Missing audit_no"):
                joined_rows(root / "results/manual_audit/audit_sheet_dingning.csv", key)
