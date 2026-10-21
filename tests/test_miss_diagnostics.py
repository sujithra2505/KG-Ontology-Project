import csv
import tempfile
import unittest
from pathlib import Path

from modules.kg_pipeline import PathRule, learn_path_rules
from modules.miss_diagnostics import diagnose_missed_triples, write_diagnostics


class MissDiagnosticsTests(unittest.TestCase):
    def setUp(self):
        self.train = [
            ("a", "r1", "m"),
            ("b", "r1", "m"),
            ("c", "r1", "m"),
            ("m", "r2", "x"),
            ("m", "r2", "y"),
            ("m", "r2", "z"),
            ("a", "r3", "x"),
            ("b", "r3", "y"),
            ("c", "r3", "z"),
        ]
        self.missed = ("a", "r3", "y")

    def diagnose(self, rules, *, limit=100, train=None):
        return diagnose_missed_triples(
            self.train if train is None else train,
            [self.missed],
            {},
            rules,
            min_support=2,
            min_confidence=0.1,
            max_rules=3000,
            max_paths_per_pivot=limit,
        )[0][0]

    def test_reports_no_two_hop_training_path(self):
        row = self.diagnose(
            [],
            train=[("a", "r1", "m"), ("m", "r2", "x")],
        )
        self.assertFalse(row["two_hop_path_exists"])
        self.assertEqual(row["diagnostic_category"], "no_two_hop_training_path")

    def test_distinguishes_rule_below_threshold_from_path_limit(self):
        row = self.diagnose(
            [],
            limit=100,
            train=[("a", "r1", "m"), ("m", "r2", "y")],
        )
        self.assertTrue(row["two_hop_path_exists"])
        self.assertFalse(row["learned_relevant_rule_exists"])
        self.assertEqual(
            row["diagnostic_category"],
            "no_qualifying_rule_even_without_path_limit",
        )

        rules, _ = learn_path_rules(
            self.train,
            min_support=2,
            min_confidence=0.1,
            max_paths_per_pivot=8,
        )
        limited_row = self.diagnose(rules, limit=8)
        self.assertTrue(limited_row["two_hop_path_exists"])
        self.assertEqual(
            limited_row["diagnostic_category"],
            "rule_qualification_lost_to_pivot_limit",
        )
        self.assertGreater(limited_row["excluded_matching_path_count"], 0)

    def test_reports_learned_rule_path_skipped_during_prediction(self):
        rule = PathRule("r1", "r2", "r3", support=2, confidence=0.5)
        row = self.diagnose([rule], limit=8)
        self.assertTrue(row["learned_relevant_rule_exists"])
        self.assertEqual(
            row["diagnostic_category"],
            "applicable_paths_excluded_by_prediction_pivot_limit",
        )
        self.assertGreater(row["excluded_applicable_path_count"], 0)

    def test_flags_learned_rule_with_eligible_path_but_no_saved_candidate(self):
        rule = PathRule("r1", "r2", "r3", support=2, confidence=0.5)
        row = self.diagnose([rule], limit=100)
        self.assertEqual(row["diagnostic_category"], "unexpected_missing_candidate")

    def test_artifact_mismatch_keeps_cause_unknown(self):
        row = diagnose_missed_triples(
            self.train,
            [self.missed],
            {},
            [],
            min_support=2,
            min_confidence=0.1,
            max_rules=3000,
            max_paths_per_pivot=100,
            artifacts_match=False,
            artifact_mismatch_reasons=["saved predictions differ"],
        )[0][0]
        self.assertEqual(
            row["diagnostic_category"],
            "unknown_saved_outputs_do_not_match_recomputation",
        )
        self.assertIn("saved predictions differ", row["evidence"])

    def test_writes_timestamped_csv_without_overwriting(self):
        row = self.diagnose([])
        with tempfile.TemporaryDirectory() as directory:
            output = write_diagnostics(Path(directory), [row])
            with output.open(encoding="utf-8", newline="") as stream:
                saved = list(csv.DictReader(stream))
            self.assertEqual(len(saved), 1)
            self.assertEqual(saved[0]["head"], self.missed[0])
            self.assertIn("diagnostic_category", saved[0])
            self.assertFalse(output.exists() and output.stat().st_size == 0)


if __name__ == "__main__":
    unittest.main()
