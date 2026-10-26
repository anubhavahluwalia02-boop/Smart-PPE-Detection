import unittest
import sys
from pathlib import Path

# Add project root to path for direct test execution
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.compliance import ComplianceAnalyzer, calculate_intersection_ratio, normalize_class_name
from src.evidence import EvidenceManager

class TestComplianceLogic(unittest.TestCase):

    def setUp(self):
        self.analyzer = ComplianceAnalyzer()
        self.evidence_mgr = EvidenceManager(cooldown_seconds=1.0)

    def test_class_normalization(self):
        self.assertEqual(normalize_class_name("hard_hat"), "helmet")
        self.assertEqual(normalize_class_name("safety-vest"), "vest")
        self.assertEqual(normalize_class_name("no_helmet"), "no_helmet")
        self.assertEqual(normalize_class_name("worker"), "person")

    def test_intersection_calculation(self):
        boxA = [0, 0, 10, 10]    # area = 100
        boxB = [5, 5, 15, 15]    # overlap = 5x5 = 25
        ratio = calculate_intersection_ratio(boxA, boxB)
        self.assertAlmostEqual(ratio, 0.25)

    def test_fully_compliant_worker(self):
        # Person bounding box from (100, 100) to (200, 400). Height = 300
        # Head is approx (100, 100) to (200, 214)
        # Torso is approx (100, 160) to (200, 334)
        detections = [
            {"box": [100, 100, 200, 400], "class_name": "person", "confidence": 0.95},
            {"box": [110, 95, 190, 150], "class_name": "helmet", "confidence": 0.92},
            {"box": [105, 170, 195, 300], "class_name": "vest", "confidence": 0.89}
        ]
        result = self.analyzer.evaluate_detections(detections, (640, 480))
        self.assertEqual(result["total_workers"], 1)
        self.assertEqual(result["frame_status"], "COMPLIANT")
        self.assertEqual(result["compliant_count"], 1)
        self.assertEqual(result["workers"][0]["status"], "COMPLIANT")

    def test_missing_helmet_violation(self):
        # Worker has vest but explicit no_helmet
        detections = [
            {"box": [100, 100, 200, 400], "class_name": "person", "confidence": 0.95},
            {"box": [115, 105, 185, 155], "class_name": "no_helmet", "confidence": 0.88},
            {"box": [105, 170, 195, 300], "class_name": "vest", "confidence": 0.90}
        ]
        result = self.analyzer.evaluate_detections(detections, (640, 480))
        self.assertEqual(result["frame_status"], "HELMET_VIOLATION")
        self.assertEqual(result["helmet_violations"], 1)
        self.assertTrue(any("Helmet" in m for m in result["workers"][0]["missing_ppe"]))

    def test_missing_vest_violation(self):
        # Worker has helmet but explicit no_vest
        detections = [
            {"box": [100, 100, 200, 400], "class_name": "person", "confidence": 0.95},
            {"box": [110, 95, 190, 150], "class_name": "helmet", "confidence": 0.92},
            {"box": [110, 180, 190, 290], "class_name": "no_vest", "confidence": 0.85}
        ]
        result = self.analyzer.evaluate_detections(detections, (640, 480))
        self.assertEqual(result["frame_status"], "SAFETY_VEST_VIOLATION")
        self.assertEqual(result["vest_violations"], 1)
        self.assertTrue(any("Vest" in m for m in result["workers"][0]["missing_ppe"]))

    def test_multiple_violations(self):
        # Worker missing both helmet and vest
        detections = [
            {"box": [100, 100, 200, 400], "class_name": "person", "confidence": 0.95},
            {"box": [115, 105, 185, 155], "class_name": "no_helmet", "confidence": 0.88},
            {"box": [110, 180, 190, 290], "class_name": "no_vest", "confidence": 0.85}
        ]
        result = self.analyzer.evaluate_detections(detections, (640, 480))
        self.assertEqual(result["frame_status"], "MULTIPLE_VIOLATIONS")
        self.assertEqual(result["multiple_violations"], 1)

    def test_partial_occlusion_needs_review(self):
        # Worker whose head is cut off by the very top boundary of the frame (py1 <= 4, small height)
        detections = [
            {"box": [100, 2, 200, 80], "class_name": "person", "confidence": 0.80}
        ]
        result = self.analyzer.evaluate_detections(detections, (640, 480))
        self.assertEqual(result["workers"][0]["status"], "NEEDS_REVIEW")

    def test_evidence_cooldown(self):
        # Test cooldown mechanism
        self.assertFalse(self.evidence_mgr.is_in_cooldown("HELMET_VIOLATION"))
        self.evidence_mgr.record_violation_timestamp("HELMET_VIOLATION")
        self.assertTrue(self.evidence_mgr.is_in_cooldown("HELMET_VIOLATION"))
        # Different violation type should not be blocked
        self.assertFalse(self.evidence_mgr.is_in_cooldown("SAFETY_VEST_VIOLATION"))

if __name__ == "__main__":
    unittest.main()
