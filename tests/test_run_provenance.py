import json
import tempfile
import unittest
from pathlib import Path

from modules.run_provenance import (
    MANIFEST_NAME,
    capture_run_snapshot,
    validate_manifest,
    verify_output_hashes,
    write_run_manifest,
)


class RunProvenanceTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        (self.root / "dataset").mkdir()
        for split in ("train", "valid", "test"):
            (self.root / "dataset" / f"{split}.txt").write_text(
                f"{split}\n", encoding="utf-8"
            )
        self.run_dir = self.root / "results" / "experiment_run"
        self.run_dir.mkdir(parents=True)
        (self.run_dir / "metrics.json").write_text(
            json.dumps({"candidate_count": 2}), encoding="utf-8"
        )
        self.snapshot = capture_run_snapshot(self.root, ())

    def tearDown(self):
        self.temporary_directory.cleanup()

    def test_manifest_has_required_fields_and_separates_settings_from_results(self):
        path = write_run_manifest(
            self.root,
            self.run_dir,
            experiment_type="test",
            run_timestamp_utc="20261021T000000Z",
            rule_settings={"min_support": 2},
            candidate_generation={"path_limit": 10},
            scoring={"method": "baseline"},
            evaluation_definition={"precision": "hits / candidates"},
            metrics={"precision": 0.25},
            used_for_learning=("train",),
            used_for_evaluation=("valid",),
            snapshot=self.snapshot,
        )
        manifest = json.loads(path.read_text(encoding="utf-8"))
        validate_manifest(manifest)

        self.assertEqual(path.name, MANIFEST_NAME)
        self.assertEqual(manifest["configured"]["rule_settings"]["min_support"], 2)
        self.assertEqual(manifest["measured"]["metrics"]["precision"], 0.25)
        self.assertTrue(
            manifest["datasets"]["splits"]["train"]["used_for_learning"]
        )
        self.assertFalse(
            manifest["datasets"]["splits"]["test"]["used_for_evaluation"]
        )
        self.assertEqual(verify_output_hashes(self.root, manifest), [])

    def test_required_schema_fields_are_enforced(self):
        with self.assertRaisesRegex(ValueError, "missing fields"):
            validate_manifest({"schema_version": 1})

    def test_output_hash_verification_reports_modified_files(self):
        path = write_run_manifest(
            self.root,
            self.run_dir,
            experiment_type="test",
            run_timestamp_utc="20261021T000000Z",
            rule_settings={},
            candidate_generation={},
            scoring={},
            evaluation_definition={},
            metrics={},
            used_for_learning=("train",),
            used_for_evaluation=("valid",),
            snapshot=self.snapshot,
        )
        manifest = json.loads(path.read_text(encoding="utf-8"))
        (self.run_dir / "metrics.json").write_text("changed", encoding="utf-8")

        self.assertEqual(
            verify_output_hashes(self.root, manifest),
            [
                "hash mismatch: results/experiment_run/metrics.json",
                "size mismatch: results/experiment_run/metrics.json",
            ],
        )

    def test_external_output_hash_is_recorded_and_verified(self):
        database = self.run_dir / "candidate.sqlite"
        database.write_bytes(b"database")
        path = write_run_manifest(
            self.root,
            self.run_dir,
            experiment_type="test",
            run_timestamp_utc="20261021T000000Z",
            rule_settings={},
            candidate_generation={},
            scoring={},
            evaluation_definition={},
            metrics={},
            used_for_learning=("train",),
            used_for_evaluation=("valid",),
            external_output_paths=(database,),
            snapshot=self.snapshot,
        )
        manifest = json.loads(path.read_text(encoding="utf-8"))

        self.assertEqual(manifest["outputs"][0]["path"], "results/experiment_run/metrics.json")
        self.assertEqual(
            manifest["external_outputs"][0]["path"],
            "results/experiment_run/candidate.sqlite",
        )
        self.assertEqual(verify_output_hashes(self.root, manifest), [])
        database.unlink()
        self.assertEqual(
            verify_output_hashes(self.root, manifest),
            ["missing output: results/experiment_run/candidate.sqlite"],
        )


if __name__ == "__main__":
    unittest.main()
