import unittest

from vesion.evaluate import UNKNOWN, Annotation, Prediction, evaluate


class EvaluationTests(unittest.TestCase):
    def test_all_point_ap_and_unknown_f1(self):
        box = (0, 0, 10, 10)
        truth = [Annotation("a", "circle", box), Annotation("b", "circle", box),
                 Annotation("c", UNKNOWN, box), Annotation("d", UNKNOWN, box)]
        predictions = [Prediction("c", "circle", box, 0.9),  # Unknown counted as a known-class FP.
                       Prediction("a", "circle", box, 0.8), Prediction("b", "circle", box, 0.7),
                       Prediction("d", UNKNOWN, box, 0.95), Prediction("e", UNKNOWN, box, 0.6)]
        result = evaluate(predictions, truth, ("circle", "square"))
        self.assertAlmostEqual(result["mAP_K"], 2 / 3)
        self.assertEqual(result["classes_without_truth"], ["square"])
        self.assertEqual(result["U-Rec"], 0.5)
        self.assertEqual(result["U-Prec"], 0.5)
        self.assertEqual(result["U-F1"], 0.5)
        self.assertAlmostEqual(result["OE-Score"], 400 / 7)

    def test_conflict_at_exact_iou_threshold_keeps_earlier_tie(self):
        truth = [Annotation("a", "circle", (0, 0, 10, 10))]
        known = Prediction("a", "circle", (0, 0, 10, 10), 0.8)
        unknown = Prediction("a", UNKNOWN, (0, 0, 5, 10), 0.8)
        for rows, expected_ap in (([known, unknown], 1.0), ([unknown, known], 0.0)):
            with self.subTest(rows=rows):
                result = evaluate(rows, truth)
                self.assertEqual(result["conflicting_predictions_removed"], 1)
                self.assertEqual(result["mAP_K"], expected_ap)

    def test_same_type_overlaps_are_kept_and_cannot_match_twice(self):
        box = (0, 0, 10, 10)
        truth = [Annotation("a", UNKNOWN, box)]
        predictions = [Prediction("a", UNKNOWN, box, 0.9), Prediction("a", UNKNOWN, box, 0.8)]
        result = evaluate(predictions, truth)
        self.assertEqual(result["conflicting_predictions_removed"], 0)
        self.assertEqual(result["unknown_counts"], {"matched": 1, "predictions": 2, "truth": 1})
        self.assertEqual(result["U-Prec"], 0.5)
        self.assertEqual(result["U-Rec"], 1.0)


if __name__ == "__main__":
    unittest.main()
