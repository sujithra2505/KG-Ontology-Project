"""Run one-factor-at-a-time rule tuning using training and validation only."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

if __package__:
    from .kg_cap_experiments import (
        DEFAULT_MAX_RULES,
        DEFAULT_MIN_CONFIDENCE,
        DEFAULT_MIN_SUPPORT,
        run_cap_experiment,
        _write_json_exclusive,
    )
    from .kg_pipeline import read_triples
    from .run_provenance import capture_run_snapshot, write_run_manifest
else:
    from kg_cap_experiments import (
        DEFAULT_MAX_RULES,
        DEFAULT_MIN_CONFIDENCE,
        DEFAULT_MIN_SUPPORT,
        run_cap_experiment,
        _write_json_exclusive,
    )
    from kg_pipeline import read_triples
    from run_provenance import capture_run_snapshot, write_run_manifest


PATH_LIMIT = 10_000
CONFIGURATIONS = (
    {"min_support": 1, "min_confidence": 0.1},
    {"min_support": 2, "min_confidence": 0.1},
    {"min_support": 4, "min_confidence": 0.1},
    {"min_support": 2, "min_confidence": 0.05},
    {"min_support": 2, "min_confidence": 0.2},
)


def _configuration_name(config: dict[str, int | float]) -> str:
    confidence = str(config["min_confidence"]).replace(".", "p")
    return f"support_{config['min_support']}_confidence_{confidence}"


def _worker(
    root: Path,
    output_dir: Path,
    config: dict[str, int | float],
) -> None:
    root = root.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    train = read_triples(root / "dataset" / "train.txt")
    validation = read_triples(root / "dataset" / "valid.txt")
    result = run_cap_experiment(
        train,
        validation,
        max_paths_per_pivot=PATH_LIMIT,
        min_support=int(config["min_support"]),
        min_confidence=float(config["min_confidence"]),
        max_rules=DEFAULT_MAX_RULES,
        database_path=output_dir / "candidates.sqlite",
    )
    validation_metrics = result["validation"]
    precision = float(validation_metrics["precision"])
    recall = float(validation_metrics["recall"])
    result["validation"]["f1"] = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    result["configuration_id"] = output_dir.name
    _write_json_exclusive(output_dir / "metrics.json", result)


def _selection(results: list[dict[str, object]]) -> dict[str, object]:
    selected = max(
        results,
        key=lambda result: (
            float(result["validation"]["f1"]),
            float(result["validation"]["precision"]),
            -int(result["prediction"]["unique_candidate_edges"]),
        ),
    )
    return {
        "configuration_id": selected["configuration_id"],
        "parameters": selected["parameters"],
        "selection_metric": "validation F1 = harmonic mean of exact-set precision and recall",
        "validation": selected["validation"],
        "tradeoff_note": (
            "Selected by validation F1 to balance precision and recall; "
            "precision breaks F1 ties, then fewer candidates. This does not "
            "establish that unmatched candidates are false."
        ),
    }


def run_tuning(root: Path) -> Path:
    root = root.resolve()
    source_files = (
        Path(__file__),
        Path(__file__).with_name("kg_cap_experiments.py"),
        Path(__file__).with_name("kg_pipeline.py"),
        Path(__file__).with_name("run_provenance.py"),
    )
    run_snapshot = capture_run_snapshot(root, source_files)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output_dir = root / "results" / "experiments" / f"rule_tuning_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=False)

    results: list[dict[str, object]] = []
    for config in CONFIGURATIONS:
        config_dir = output_dir / _configuration_name(config)
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--worker",
            "--root",
            str(root),
            "--output-dir",
            str(config_dir),
            "--min-support",
            str(config["min_support"]),
            "--min-confidence",
            str(config["min_confidence"]),
        ]
        subprocess.run(command, check=True, capture_output=True, text=True)
        result = json.loads(
            (config_dir / "metrics.json").read_text(encoding="utf-8")
        )
        results.append(result)

    baseline = next(
        result
        for result in results
        if result["parameters"]["min_support"] == DEFAULT_MIN_SUPPORT
        and result["parameters"]["min_confidence"] == DEFAULT_MIN_CONFIDENCE
    )
    summary = {
        "run_timestamp_utc": timestamp,
        "dataset_files_used": ["dataset/train.txt", "dataset/valid.txt"],
        "test_split_used": False,
        "shared_parameters": {
            "max_rules": DEFAULT_MAX_RULES,
            "max_paths_per_pivot": PATH_LIMIT,
        },
        "experiment_design": {
            "support_sweep": {
                "confidence_fixed": DEFAULT_MIN_CONFIDENCE,
                "values": [1, 2, 4],
            },
            "confidence_sweep": {
                "support_fixed": DEFAULT_MIN_SUPPORT,
                "values": [0.05, 0.1, 0.2],
            },
            "max_rules_reached": any(
                result["learned_rules"] >= DEFAULT_MAX_RULES
                for result in results
            ),
        },
        "commands": [
            "py -3.13 modules\\kg_rule_tuning.py",
            "py -3.13 modules\\kg_rule_tuning.py --worker --root <project-root> --output-dir <new-dir> --min-support <n> --min-confidence <x>",
        ],
        "baseline_configuration_id": baseline["configuration_id"],
        "baseline": {
            "candidate_count": baseline["prediction"]["unique_candidate_edges"],
            "learned_rules": baseline["learned_rules"],
            "validation": baseline["validation"],
        },
        "selection": _selection(results),
        "results": results,
    }
    _write_json_exclusive(output_dir / "summary.json", summary)
    write_run_manifest(
        root,
        output_dir,
        experiment_type="validation_rule_tuning",
        run_timestamp_utc=timestamp,
        rule_settings={
            "support_sweep": summary["experiment_design"]["support_sweep"],
            "confidence_sweep": summary["experiment_design"]["confidence_sweep"],
            "max_rules": DEFAULT_MAX_RULES,
            "max_paths_per_pivot": PATH_LIMIT,
        },
        candidate_generation={
            "candidate_generation_parameters_vary_by_configuration": True,
            "configuration_ids": [
                result["configuration_id"] for result in results
            ],
        },
        scoring={"method": "not scored; candidate membership only"},
        evaluation_definition={
            "precision": (
                "Unique candidate/validation exact triple matches divided by "
                "unique generated candidates."
            ),
            "recall": (
                "Unique candidate/validation exact triple matches divided by "
                "unique known validation triples."
            ),
            "selection": summary["selection"]["selection_metric"],
            "unmatched_candidates": "Unknown truth; not treated as false.",
        },
        metrics={
            "baseline": summary["baseline"],
            "selection": summary["selection"],
            "configurations": results,
        },
        used_for_learning=("train",),
        used_for_evaluation=("valid",),
        snapshot=run_snapshot,
    )
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Tune rule support/confidence using validation only."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Project root containing dataset and results directories.",
    )
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--output-dir", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--min-support", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--min-confidence", type=float, help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.worker:
        if (
            args.output_dir is None
            or args.min_support is None
            or args.min_confidence is None
        ):
            parser.error("worker mode requires output directory and both thresholds")
        _worker(
            args.root,
            args.output_dir,
            {
                "min_support": args.min_support,
                "min_confidence": args.min_confidence,
            },
        )
        return

    output_dir = run_tuning(args.root)
    print(json.dumps({"output_dir": str(output_dir)}, indent=2))


if __name__ == "__main__":
    main()
