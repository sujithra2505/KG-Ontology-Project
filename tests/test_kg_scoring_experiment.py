import sqlite3
import tempfile
import unittest
from pathlib import Path

from modules.kg_cap_experiments import _validation_metrics
from modules.kg_pipeline import PathRule
from modules.kg_scoring_experiment import (
    _compute_combined_scores,
    combine_distinct_rule_confidences,
    generate_scoring_evidence,
)


class CandidateScoringExperimentTests(unittest.TestCase):
    def test_noisy_or_uses_distinct_rules_once(self):
        rule_a = ("r1", "r2", "target")
        rule_b = ("r3", "r4", "target")
        score = combine_distinct_rule_confidences(
            [
                (rule_a, 0.5),
                (rule_a, 0.5),
                (rule_b, 0.5),
            ]
        )
        self.assertAlmostEqual(score, 0.75)
        self.assertAlmostEqual(
            combine_distinct_rule_confidences([(rule_a, 0.5)]), 0.5
        )

    def test_repeated_paths_from_same_rule_do_not_duplicate_evidence(self):
        train = [
            ("h", "r1", "m1"),
            ("h", "r1", "m2"),
            ("m1", "r2", "t"),
            ("m2", "r2", "t"),
            ("h", "r3", "m3"),
            ("m3", "r4", "t"),
        ]
        rules = [
            PathRule("r1", "r2", "target", support=2, confidence=0.5),
            PathRule("r3", "r4", "target", support=2, confidence=0.5),
        ]
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "scores.sqlite"
            stats = generate_scoring_evidence(train, rules, 100, database)
            score_stats = _compute_combined_scores(database)
            validation = [("h", "target", "t")]
            baseline = _validation_metrics(
                database, validation, stats["unique_candidate_edges"]
            )
            combined = _validation_metrics(
                database,
                validation,
                stats["unique_candidate_edges"],
                candidate_table="combined_scores",
            )
            connection = sqlite3.connect(database)
            try:
                evidence_count = connection.execute(
                    "SELECT COUNT(*) FROM rule_evidence "
                    "WHERE head='h' AND relation='target' AND tail='t'"
                ).fetchone()[0]
                scores = connection.execute(
                    "SELECT c.baseline_score, s.score FROM candidates c "
                    "JOIN combined_scores s USING (head, relation, tail) "
                    "WHERE c.head='h' AND c.relation='target' AND c.tail='t'"
                ).fetchone()
            finally:
                connection.close()

        self.assertEqual(stats["unique_candidate_edges"], 1)
        self.assertEqual(evidence_count, 2)
        self.assertAlmostEqual(scores[0], 0.5)
        self.assertAlmostEqual(scores[1], 0.75)
        self.assertEqual(baseline["predicted_edges_found"], 1)
        self.assertEqual(combined["predicted_edges_found"], 1)
        self.assertEqual(baseline["precision"], combined["precision"])
        self.assertEqual(baseline["recall"], combined["recall"])
        self.assertEqual(score_stats["candidates_with_changed_score"], 1)

    def test_confidence_validation_and_existing_database_are_enforced(self):
        with self.assertRaises(ValueError):
            combine_distinct_rule_confidences([(("r1", "r2", "r3"), 1.1)])
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "existing.sqlite"
            database.write_text("preserve", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                generate_scoring_evidence([], [], 10, database)
            self.assertEqual(database.read_text(encoding="utf-8"), "preserve")


if __name__ == "__main__":
    unittest.main()
