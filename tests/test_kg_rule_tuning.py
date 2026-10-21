import unittest

from modules.kg_rule_tuning import CONFIGURATIONS, _selection


class RuleTuningTests(unittest.TestCase):
    def test_configurations_change_one_threshold_at_a_time(self):
        baseline = next(
            config
            for config in CONFIGURATIONS
            if config["min_support"] == 2
            and config["min_confidence"] == 0.1
        )
        support_sweep = [
            config for config in CONFIGURATIONS if config["min_confidence"] == 0.1
        ]
        confidence_sweep = [
            config for config in CONFIGURATIONS if config["min_support"] == 2
        ]

        self.assertEqual({config["min_support"] for config in support_sweep}, {1, 2, 4})
        self.assertTrue(
            all(config["min_confidence"] == baseline["min_confidence"] for config in support_sweep)
        )
        self.assertEqual(
            {config["min_confidence"] for config in confidence_sweep},
            {0.05, 0.1, 0.2},
        )
        self.assertTrue(
            all(config["min_support"] == baseline["min_support"] for config in confidence_sweep)
        )

    def test_selection_uses_f1_then_precision_then_candidate_count(self):
        results = [
            {
                "configuration_id": "more_recall",
                "parameters": {"min_support": 1},
                "prediction": {"unique_candidate_edges": 100},
                "validation": {"f1": 0.4, "precision": 0.2, "recall": 0.8},
            },
            {
                "configuration_id": "best_balance",
                "parameters": {"min_support": 2},
                "prediction": {"unique_candidate_edges": 50},
                "validation": {"f1": 0.5, "precision": 0.4, "recall": 0.7},
            },
        ]
        selected = _selection(results)
        self.assertEqual(selected["configuration_id"], "best_balance")
        self.assertIn("validation F1", selected["selection_metric"])

    def test_selection_breaks_ties_toward_precision_then_fewer_candidates(self):
        results = [
            {
                "configuration_id": "low_precision",
                "parameters": {},
                "prediction": {"unique_candidate_edges": 10},
                "validation": {"f1": 0.5, "precision": 0.3},
            },
            {
                "configuration_id": "high_precision",
                "parameters": {},
                "prediction": {"unique_candidate_edges": 30},
                "validation": {"f1": 0.5, "precision": 0.4},
            },
            {
                "configuration_id": "fewer_candidates",
                "parameters": {},
                "prediction": {"unique_candidate_edges": 20},
                "validation": {"f1": 0.5, "precision": 0.4},
            },
        ]
        self.assertEqual(_selection(results)["configuration_id"], "fewer_candidates")


if __name__ == "__main__":
    unittest.main()
