import unittest

from modules.kg_pipeline import _evaluate_predictions


class ExactSetEvaluationTests(unittest.TestCase):
    def test_precision_and_recall_use_documented_split_denominators(self):
        validation_match = ("v_head", "r", "v_tail")
        validation_missed = ("v2_head", "r", "v2_tail")
        test_match = ("t_head", "r", "t_tail")
        unmatched_candidate = ("u_head", "r", "u_tail")
        predictions = {
            validation_match: (0.9, 3),
            test_match: (0.8, 2),
            unmatched_candidate: (0.7, 1),
        }

        metrics = _evaluate_predictions(
            predictions,
            set(),
            {validation_match, validation_missed},
            {test_match},
        )

        validation = metrics["validation"]
        test = metrics["test"]
        self.assertEqual(validation["known_edges"], 2)
        self.assertEqual(validation["predicted_edges_found"], 1)
        self.assertAlmostEqual(validation["precision"], 1 / 3)
        self.assertAlmostEqual(validation["recall"], 1 / 2)
        self.assertEqual(test["known_edges"], 1)
        self.assertEqual(test["predicted_edges_found"], 1)
        self.assertAlmostEqual(test["precision"], 1 / 3)
        self.assertAlmostEqual(test["recall"], 1)
        self.assertIn("not confirmed false", metrics["note"])

    def test_empty_candidate_or_split_denominators_produce_zero(self):
        empty_candidates = _evaluate_predictions(
            {},
            set(),
            {("h", "r", "t")},
            set(),
        )
        self.assertEqual(empty_candidates["validation"]["precision"], 0)
        self.assertEqual(empty_candidates["validation"]["recall"], 0)
        self.assertEqual(empty_candidates["test"]["precision"], 0)
        self.assertEqual(empty_candidates["test"]["recall"], 0)

        empty_splits = _evaluate_predictions(
            {("h", "r", "t"): (0.5, 1)},
            set(),
            set(),
            set(),
        )
        self.assertEqual(empty_splits["validation"]["precision"], 0)
        self.assertEqual(empty_splits["validation"]["recall"], 0)
        self.assertEqual(empty_splits["test"]["precision"], 0)
        self.assertEqual(empty_splits["test"]["recall"], 0)


if __name__ == "__main__":
    unittest.main()
