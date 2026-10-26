import unittest
import sys
from pathlib import Path
import tempfile

# Add project root to path for direct test execution
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.database import init_db, log_violation, get_violations, get_dashboard_stats, delete_violation

class TestDatabase(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temp_dir.name) / "test_ppe.db"
        init_db(self.db_path)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_insert_and_retrieve(self):
        rec_id = log_violation(
            violation_type="HELMET_VIOLATION",
            missing_ppe="Helmet",
            confidence=0.88,
            source="Test Camera",
            evidence_path="violations/test1.jpg",
            person_count=1,
            notes="No helmet found on worker #1",
            db_path=self.db_path
        )
        self.assertIsNotNone(rec_id)
        records = get_violations(limit=10, db_path=self.db_path)
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["violation_type"], "HELMET_VIOLATION")
        self.assertEqual(records[0]["missing_ppe"], "Helmet")

    def test_filter_by_violation_type(self):
        log_violation("HELMET_VIOLATION", "Helmet", 0.9, "cam", db_path=self.db_path)
        log_violation("SAFETY_VEST_VIOLATION", "Safety Vest", 0.85, "cam", db_path=self.db_path)
        log_violation("MULTIPLE_VIOLATIONS", "Helmet, Vest", 0.95, "cam", db_path=self.db_path)

        helmet_only = get_violations(filter_type="HELMET_VIOLATION", db_path=self.db_path)
        self.assertEqual(len(helmet_only), 1)
        self.assertEqual(helmet_only[0]["violation_type"], "HELMET_VIOLATION")

    def test_dashboard_stats(self):
        log_violation("HELMET_VIOLATION", "Helmet", 0.9, "cam", db_path=self.db_path)
        log_violation("HELMET_VIOLATION", "Helmet", 0.85, "cam", db_path=self.db_path)
        log_violation("SAFETY_VEST_VIOLATION", "Safety Vest", 0.92, "cam", db_path=self.db_path)

        stats = get_dashboard_stats(self.db_path)
        self.assertEqual(stats["total_violations"], 3)
        self.assertEqual(stats["helmet_violations"], 2)
        self.assertEqual(stats["vest_violations"], 1)

    def test_delete_record(self):
        rec_id = log_violation("HELMET_VIOLATION", "Helmet", 0.9, "cam", db_path=self.db_path)
        self.assertTrue(delete_violation(rec_id, self.db_path))
        records = get_violations(limit=10, db_path=self.db_path)
        self.assertEqual(len(records), 0)

if __name__ == "__main__":
    unittest.main()
