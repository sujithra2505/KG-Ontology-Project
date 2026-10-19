import tempfile
import unittest
from pathlib import Path

from rdflib import OWL, RDF, RDFS

from modules.kg_pipeline import (
    build_knowledge_graph,
    build_ontology,
    learn_path_rules,
    predict_missing_edges,
    read_triples,
    write_prediction_graph,
)
from modules.kg_viewer import write_full_graph_viewer
from modules.prediction_viewer import write_prediction_explorer


class KnowledgeGraphPipelineTests(unittest.TestCase):
    def setUp(self):
        self.triples = [
            ("alice", "acts_in", "film_1"),
            ("film_1", "has_genre", "drama"),
            ("alice", "person_genre", "drama"),
            ("bob", "acts_in", "film_2"),
            ("film_2", "has_genre", "comedy"),
            ("bob", "person_genre", "comedy"),
            ("carol", "acts_in", "film_3"),
            ("film_3", "has_genre", "drama"),
        ]

    def test_graph_preserves_direction_and_parallel_relations(self):
        graph = build_knowledge_graph(
            [("alice", "acts_in", "film"), ("alice", "produced", "film")]
        )
        self.assertEqual(graph.number_of_nodes(), 2)
        self.assertEqual(graph.number_of_edges(), 2)
        self.assertEqual(
            {data["relation"] for data in graph.get_edge_data("alice", "film").values()},
            {"acts_in", "produced"},
        )

    def test_ontology_contains_schema_and_truth_assertions(self):
        ontology = build_ontology(self.triples[:2])
        properties = list(ontology.subjects(RDF.type, OWL.ObjectProperty))
        self.assertEqual(len(properties), 2)
        self.assertTrue(list(ontology.triples((None, RDFS.domain, None))))
        self.assertTrue(list(ontology.triples((None, RDFS.range, None))))
        self.assertTrue(
            any(str(predicate).endswith("acts_in") for _, predicate, _ in ontology)
        )

    def test_rules_predict_only_novel_edges(self):
        rules, _ = learn_path_rules(
            self.triples, min_support=2, min_confidence=0.5
        )
        self.assertIn(
            ("acts_in", "has_genre", "person_genre"),
            {
                (r.first_relation, r.second_relation, r.predicted_relation)
                for r in rules
            },
        )
        predictions, _ = predict_missing_edges(self.triples, rules)
        self.assertIn(("carol", "person_genre", "drama"), predictions)
        self.assertNotIn(("alice", "person_genre", "drama"), predictions)
        self.assertNotIn(("bob", "person_genre", "comedy"), predictions)

    def test_path_limit_skips_high_degree_pivots(self):
        rules, stats = learn_path_rules(
            [
                ("alice", "acts_in", "film_1"),
                ("film_1", "has_genre", "drama"),
                ("film_1", "has_language", "english"),
            ],
            min_support=1,
            min_confidence=0,
            max_paths_per_pivot=1,
        )
        self.assertTrue(all(rule.support > 0 for rule in rules))
        self.assertGreater(stats["skipped_high_degree_pivots"], 0)
        self.assertGreater(stats["skipped_two_hop_paths"], 0)

    def test_read_triples_rejects_malformed_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.tsv"
            path.write_text("head\trelation\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "expected three"):
                read_triples(path)

    def test_prediction_graph_shows_held_out_confirmation(self):
        rules, _ = learn_path_rules(
            self.triples, min_support=2, min_confidence=0.5
        )
        predictions, _ = predict_missing_edges(self.triples, rules)
        heldout = [("carol", "person_genre", "drama")]
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "example.svg"
            write_prediction_graph(
                output, self.triples, predictions, [], heldout, rules
            )
            svg = output.read_text(encoding="utf-8")
        self.assertIn("MATCHES TEST DATA", svg)
        self.assertIn("carol", svg)
        self.assertIn("person_genre", svg)
        self.assertIn("Observed supporting training facts", svg)

    def test_full_graph_viewer_embeds_all_graph_data(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "full_graph.html"
            write_full_graph_viewer(self.triples, output)
            html = output.read_text(encoding="utf-8")
        self.assertIn("Complete knowledge graph", html)
        self.assertIn("const DATA = ", html)
        self.assertIn("Search an entity", html)
        self.assertIn("Show all", html)

    def test_prediction_explorer_embeds_predictions_and_support_rules(self):
        rules, _ = learn_path_rules(
            self.triples, min_support=2, min_confidence=0.5
        )
        predictions, _ = predict_missing_edges(self.triples, rules)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "predictions.html"
            write_prediction_explorer(
                self.triples,
                predictions,
                rules,
                [],
                [("carol", "person_genre", "drama")],
                output,
            )
            html = output.read_text(encoding="utf-8")
        self.assertIn("Predicted edges and their supporting paths", html)
        self.assertIn("No held-out match", html)
        self.assertIn("const DATA=", html)
        self.assertIn('"carol"', html)


if __name__ == "__main__":
    unittest.main()
