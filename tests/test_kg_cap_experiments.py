import sqlite3
import tempfile
import unittest
from pathlib import Path

from modules.kg_cap_experiments import (
    _validation_metrics,
    stream_candidates_to_sqlite,
)
from modules.kg_pipeline import learn_path_rules, predict_missing_edges


class PathCapExperimentTests(unittest.TestCase):
    def setUp(self):
        self.train = [
            ("alice", "acted_in", "film_1"),
            ("film_1", "has_genre", "drama"),
            ("alice", "person_genre", "drama"),
            ("bob", "acted_in", "film_2"),
            ("film_2", "has_genre", "drama"),
        ]
        self.validation = [
            ("bob", "person_genre", "drama"),
            ("alice", "person_genre", "drama"),
        ]

    def test_disk_stream_candidates_match_in_memory_baseline(self):
        rules, _ = learn_path_rules(
            self.train,
            min_support=1,
            min_confidence=0,
            max_paths_per_pivot=100,
        )
        baseline, baseline_stats = predict_missing_edges(
            self.train, rules, max_paths_per_pivot=100
        )
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "candidates.sqlite"
            stats = stream_candidates_to_sqlite(
                self.train, rules, 100, database
            )
            connection = sqlite3.connect(database)
            try:
                actual = set(
                    connection.execute(
                        "SELECT head, relation, tail FROM candidates"
                    )
                )
            finally:
                connection.close()

        self.assertEqual(actual, set(baseline))
        self.assertEqual(stats["unique_candidate_edges"], len(baseline))
        self.assertEqual(
            stats["skipped_high_degree_pivots"],
            baseline_stats["skipped_high_degree_pivots"],
        )
        self.assertEqual(
            stats["skipped_two_hop_paths"],
            baseline_stats["skipped_two_hop_paths"],
        )

    def test_disk_evaluation_uses_unique_candidate_set_definition(self):
        rules, _ = learn_path_rules(
            self.train,
            min_support=1,
            min_confidence=0,
            max_paths_per_pivot=100,
        )
        predictions, _ = predict_missing_edges(
            self.train, rules, max_paths_per_pivot=100
        )
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "candidates.sqlite"
            stream_candidates_to_sqlite(self.train, rules, 100, database)
            actual = _validation_metrics(database, self.validation, len(predictions))

        hits = len(set(predictions) & set(self.validation))
        self.assertEqual(actual["predicted_edges_found"], hits)
        self.assertEqual(actual["known_edges"], len(set(self.validation)))
        self.assertAlmostEqual(actual["precision"], hits / len(predictions))
        self.assertAlmostEqual(actual["recall"], hits / len(set(self.validation)))

    def test_candidate_database_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "exists.sqlite"
            database.write_text("preserve", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                stream_candidates_to_sqlite(self.train, [], 100, database)
            self.assertEqual(database.read_text(encoding="utf-8"), "preserve")


if __name__ == "__main__":
    unittest.main()
