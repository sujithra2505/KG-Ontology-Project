"""Explain missed held-out triples using training-only path evidence."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

if __package__:
    from .kg_pipeline import (
        PathRule,
        Triple,
        learn_path_rules,
        predict_missing_edges,
        read_triples,
    )
    from .evaluation_dashboard import read_predictions
    from .run_provenance import capture_run_snapshot, write_run_manifest
else:
    from kg_pipeline import (
        PathRule,
        Triple,
        learn_path_rules,
        predict_missing_edges,
        read_triples,
    )
    from evaluation_dashboard import read_predictions
    from run_provenance import capture_run_snapshot, write_run_manifest


PATH_EXAMPLE_LIMIT = 10


@dataclass
class _PathEvidence:
    count: int = 0
    examples: list[dict[str, object]] = field(default_factory=list)
    body_path_counts: Counter[tuple[str, str]] = field(default_factory=Counter)
    excluded_body_path_counts: Counter[tuple[str, str]] = field(
        default_factory=Counter
    )
    excluded_pivots_by_body: dict[tuple[str, str], set[str]] = field(
        default_factory=lambda: defaultdict(set)
    )


def _rule_key(rule: PathRule) -> tuple[str, str, str]:
    return (rule.first_relation, rule.second_relation, rule.predicted_relation)


def _count_training_paths(
    train: list[Triple],
    missed: set[Triple],
    max_paths_per_pivot: int,
) -> tuple[
    Counter[tuple[str, str]],
    Counter[tuple[tuple[str, str], str]],
    Counter[tuple[str, str]],
    Counter[tuple[tuple[str, str], str]],
    dict[Triple, _PathEvidence],
    int,
    int,
]:
    outgoing: dict[str, list[tuple[str, str]]] = defaultdict(list)
    incoming: dict[str, list[tuple[str, str]]] = defaultdict(list)
    relations_by_pair: dict[tuple[str, str], set[str]] = defaultdict(set)
    missed_by_pair: dict[tuple[str, str], list[Triple]] = defaultdict(list)

    for head, relation, tail in train:
        outgoing[head].append((relation, tail))
        incoming[tail].append((relation, head))
        relations_by_pair[(head, tail)].add(relation)
    for triple in missed:
        missed_by_pair[(triple[0], triple[2])].append(triple)

    full_body_support: Counter[tuple[str, str]] = Counter()
    full_rule_support: Counter[tuple[tuple[str, str], str]] = Counter()
    limited_body_support: Counter[tuple[str, str]] = Counter()
    limited_rule_support: Counter[tuple[tuple[str, str], str]] = Counter()
    path_evidence = {triple: _PathEvidence() for triple in missed}
    skipped_pivot_count = 0
    skipped_path_count = 0

    for pivot, incoming_edges in incoming.items():
        outgoing_edges = outgoing.get(pivot, ())
        possible_paths = len(incoming_edges) * len(outgoing_edges)
        exceeds_limit = possible_paths > max_paths_per_pivot
        if exceeds_limit:
            skipped_pivot_count += 1
            skipped_path_count += possible_paths

        for first_relation, head in incoming_edges:
            for second_relation, tail in outgoing_edges:
                body = (first_relation, second_relation)
                full_body_support[body] += 1
                if not exceeds_limit:
                    limited_body_support[body] += 1
                for target_relation in relations_by_pair.get((head, tail), ()):
                    rule_key = (body, target_relation)
                    full_rule_support[rule_key] += 1
                    if not exceeds_limit:
                        limited_rule_support[rule_key] += 1

                for triple in missed_by_pair.get((head, tail), ()):
                    evidence = path_evidence[triple]
                    evidence.count += 1
                    evidence.body_path_counts[body] += 1
                    if exceeds_limit:
                        evidence.excluded_body_path_counts[body] += 1
                        evidence.excluded_pivots_by_body[body].add(pivot)
                    examples = evidence.examples
                    if len(examples) < PATH_EXAMPLE_LIMIT:
                        examples.append(
                            {
                                "middle": pivot,
                                "first_relation": first_relation,
                                "second_relation": second_relation,
                                "pivot_limit_excluded": exceeds_limit,
                            }
                        )

    return (
        full_body_support,
        full_rule_support,
        limited_body_support,
        limited_rule_support,
        path_evidence,
        skipped_pivot_count,
        skipped_path_count,
    )


def _qualified_rules(
    body_support: Counter[tuple[str, str]],
    rule_support: Counter[tuple[tuple[str, str], str]],
    min_support: int,
    min_confidence: float,
) -> set[tuple[str, str, str]]:
    return {
        (body[0], body[1], target)
        for (body, target), support in rule_support.items()
        if support >= min_support
        and support / body_support[body] >= min_confidence
    }


def _rules_from_file(path: Path) -> list[PathRule]:
    rules: list[PathRule] = []
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        expected = {
            "body_relation_1",
            "body_relation_2",
            "predicted_relation",
            "support",
            "confidence",
        }
        if reader.fieldnames is None or not expected.issubset(reader.fieldnames):
            raise ValueError(f"{path}: missing required learned-rule columns")
        for line_number, row in enumerate(reader, start=2):
            if any(row.get(column) is None for column in expected):
                raise ValueError(f"{path}:{line_number}: incomplete rule row")
            rules.append(
                PathRule(
                    first_relation=row["body_relation_1"],
                    second_relation=row["body_relation_2"],
                    predicted_relation=row["predicted_relation"],
                    support=int(row["support"]),
                    confidence=float(row["confidence"]),
                )
            )
    return rules


def _prediction_artifacts_match(
    train: list[Triple],
    saved_predictions: dict[Triple, tuple[float, int]],
    saved_rules: list[PathRule],
    metrics: dict[str, object],
    params: dict[str, object],
) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    try:
        min_support = int(params["min_support"])
        min_confidence = float(params["min_confidence"])
        max_rules = int(params["max_rules"])
        max_paths_per_pivot = int(params["max_paths_per_pivot"])
        expected_rules, mining_stats = learn_path_rules(
            train,
            min_support=min_support,
            min_confidence=min_confidence,
            max_rules=max_rules,
            max_paths_per_pivot=max_paths_per_pivot,
        )
        expected_predictions, prediction_stats = predict_missing_edges(
            train, expected_rules, max_paths_per_pivot=max_paths_per_pivot
        )
    except (KeyError, TypeError, ValueError) as error:
        return False, [f"could not recompute using saved parameters: {error}"]

    def rule_map(rules: Iterable[PathRule]) -> dict[tuple[str, str, str], PathRule]:
        return {_rule_key(rule): rule for rule in rules}

    expected_rule_map = rule_map(expected_rules)
    saved_rule_map = rule_map(saved_rules)
    if expected_rule_map.keys() != saved_rule_map.keys() or any(
        expected_rule_map[key].support != saved_rule_map[key].support
        or abs(expected_rule_map[key].confidence - saved_rule_map[key].confidence)
        > 1e-7
        for key in expected_rule_map.keys() & saved_rule_map.keys()
    ):
        reasons.append("learned_rules.tsv does not match rules recomputed from train")

    if expected_predictions.keys() != saved_predictions.keys():
        reasons.append(
            "predicted_edges.tsv candidates do not match candidates recomputed from train"
        )
    else:
        for triple, (confidence, support) in expected_predictions.items():
            saved_confidence, saved_support = saved_predictions[triple]
            if support != saved_support or abs(confidence - saved_confidence) > 5.1e-9:
                reasons.append(
                    "predicted_edges.tsv scores do not match recomputed candidates"
                )
                break

    if metrics.get("learned_rules") != len(expected_rules):
        reasons.append("metrics.json learned_rules count differs from recomputed rules")
    mining = metrics.get("mining")
    prediction = metrics.get("prediction")
    if not isinstance(mining, dict) or any(
        mining.get(key) != value for key, value in mining_stats.items()
    ):
        reasons.append("metrics.json mining statistics differ from recomputed rules")
    if not isinstance(prediction, dict) or any(
        prediction.get(key) != value for key, value in prediction_stats.items()
    ):
        reasons.append("metrics.json prediction statistics differ from recomputation")
    if metrics.get("predicted_unique_edges") != len(expected_predictions):
        reasons.append("metrics.json candidate count differs from recomputation")
    if metrics.get("training_edges") != len(train):
        reasons.append("metrics.json training edge count differs from train.txt")

    return not reasons, reasons


def diagnose_missed_triples(
    train: list[Triple],
    test: list[Triple],
    predictions: dict[Triple, tuple[float, int]],
    learned_rules: list[PathRule],
    *,
    min_support: int,
    min_confidence: float,
    max_rules: int,
    max_paths_per_pivot: int,
    artifacts_match: bool = True,
    artifact_mismatch_reasons: list[str] | None = None,
) -> tuple[list[dict[str, object]], dict[str, int]]:
    """Classify every test triple absent from candidates using training evidence."""
    missed_rows = [triple for triple in test if triple not in predictions]
    missed_set = set(missed_rows)
    (
        full_body_support,
        full_rule_support,
        limited_body_support,
        limited_rule_support,
        path_evidence,
        skipped_pivot_count,
        skipped_path_count,
    ) = _count_training_paths(train, missed_set, max_paths_per_pivot)

    full_qualified = _qualified_rules(
        full_body_support, full_rule_support, min_support, min_confidence
    )
    limited_qualified = _qualified_rules(
        limited_body_support, limited_rule_support, min_support, min_confidence
    )
    learned_rule_keys = {_rule_key(rule) for rule in learned_rules}
    learned_by_key = {_rule_key(rule): rule for rule in learned_rules}

    diagnostics: list[dict[str, object]] = []
    category_counts: Counter[str] = Counter()
    for head, target_relation, tail in missed_rows:
        triple = (head, target_relation, tail)
        evidence = path_evidence[triple]
        path_count = evidence.count
        examples = evidence.examples
        bodies = set(evidence.body_path_counts)
        relevant_keys = {
            (body[0], body[1], target_relation) for body in bodies
        }
        learned_keys = relevant_keys & learned_rule_keys
        full_keys = relevant_keys & full_qualified
        limited_keys = relevant_keys & limited_qualified

        learned_bodies = {(key[0], key[1]) for key in learned_keys}
        eligible_learned_path_count = sum(
            evidence.body_path_counts[body]
            - evidence.excluded_body_path_counts[body]
            for body in learned_bodies
        )
        excluded_path_count = sum(
            evidence.excluded_body_path_counts[body] for body in learned_bodies
        )

        if not artifacts_match:
            category = "unknown_saved_outputs_do_not_match_recomputation"
        elif path_count == 0:
            category = "no_two_hop_training_path"
        elif learned_keys and eligible_learned_path_count:
            category = "unexpected_missing_candidate"
        elif learned_keys and not eligible_learned_path_count:
            category = "applicable_paths_excluded_by_prediction_pivot_limit"
        elif full_keys - limited_keys:
            category = "rule_qualification_lost_to_pivot_limit"
        elif limited_keys and not learned_keys and len(limited_qualified) > max_rules:
            category = "rule_excluded_by_max_rules_limit"
        elif full_keys and limited_keys and not learned_keys:
            category = "qualified_rule_missing_from_saved_rule_artifact"
        elif not full_keys:
            category = "no_qualifying_rule_even_without_path_limit"
        else:
            category = "cause_unknown"

        rule_details = []
        for key in sorted(relevant_keys):
            body = (key[0], key[1])
            full_support = full_rule_support[(body, target_relation)]
            capped_support = limited_rule_support[(body, target_relation)]
            full_confidence = (
                full_support / full_body_support[body]
                if full_body_support[body]
                else 0.0
            )
            capped_confidence = (
                capped_support / limited_body_support[body]
                if limited_body_support[body]
                else 0.0
            )
            learned = learned_by_key.get(key)
            rule_details.append(
                {
                    "first_relation": key[0],
                    "second_relation": key[1],
                    "target_relation": key[2],
                    "learned": learned is not None,
                    "learned_support": learned.support if learned else None,
                    "learned_confidence": learned.confidence if learned else None,
                    "full_graph_support": full_support,
                    "full_graph_confidence": full_confidence,
                    "within_limit_support": capped_support,
                    "within_limit_confidence": capped_confidence,
                }
            )

        excluded_pivots = sorted(
            {
                pivot
                for body in evidence.excluded_pivots_by_body
                for pivot in evidence.excluded_pivots_by_body[body]
            }
        )
        excluded_matching_path_count = sum(
            evidence.excluded_body_path_counts.values()
        )
        evidence_parts = [
            f"{path_count} matching two-hop training path(s) found by unrestricted inspection.",
            (
                f"{excluded_matching_path_count} matching path(s) at skipped pivots, "
                f"of which {excluded_path_count} use learned target rules; "
                f"{len(excluded_pivots)} matching pivot(s) exceeded the configured "
                f"{max_paths_per_pivot}-path limit."
                if excluded_matching_path_count
                else "No matching path was found at an over-limit pivot."
            ),
        ]
        if artifact_mismatch_reasons:
            evidence_parts.extend(artifact_mismatch_reasons)
        if category == "no_qualifying_rule_even_without_path_limit":
            evidence_parts.append(
                "No matching path-body/target rule meets the saved support and "
                "confidence thresholds even when all training pivots are counted."
            )
        elif category == "rule_qualification_lost_to_pivot_limit":
            evidence_parts.append(
                "At least one matching body/target rule qualifies with all paths "
                "counted but not with the configured pivot cutoff for that same rule."
            )
        elif category == "unexpected_missing_candidate":
            evidence_parts.append(
                "A learned target rule and an eligible supporting path exist; "
                "the saved candidate output is inconsistent with recomputation."
            )

        category_counts[category] += 1
        diagnostics.append(
            {
                "head": head,
                "relation": target_relation,
                "tail": tail,
                "diagnostic_category": category,
                "two_hop_path_exists": path_count > 0,
                "matching_path_count": path_count,
                "path_examples_json": json.dumps(
                    examples, ensure_ascii=True, separators=(",", ":")
                ),
                "path_examples_truncated": path_count > len(examples),
                "learned_relevant_rule_exists": bool(learned_keys),
                "relevant_rules_json": json.dumps(
                    rule_details, ensure_ascii=True, separators=(",", ":")
                ),
                "excluded_matching_path_count": excluded_matching_path_count,
                "excluded_applicable_path_count": excluded_path_count,
                "excluded_pivots_json": json.dumps(
                    excluded_pivots, ensure_ascii=True, separators=(",", ":")
                ),
                "path_limit": max_paths_per_pivot,
                "evidence": " ".join(evidence_parts),
            }
        )

    return diagnostics, {
        "missed_test_triples": len(diagnostics),
        "skipped_high_degree_pivots": skipped_pivot_count,
        "skipped_two_hop_paths": skipped_path_count,
        **{f"category_{key}": value for key, value in sorted(category_counts.items())},
    }


def analyze_project(root: Path) -> tuple[list[dict[str, object]], dict[str, int]]:
    root = root.resolve()
    dataset_dir = root / "dataset"
    results_dir = root / "results"
    train = read_triples(dataset_dir / "train.txt")
    test = read_triples(dataset_dir / "test.txt")
    predictions = read_predictions(results_dir / "predicted_edges.tsv")
    learned_rules = _rules_from_file(results_dir / "learned_rules.tsv")
    metrics = json.loads((results_dir / "metrics.json").read_text(encoding="utf-8"))
    params = metrics.get("parameters")
    if not isinstance(params, dict):
        raise ValueError("results/metrics.json is missing saved rule parameters")
    artifacts_match, mismatch_reasons = _prediction_artifacts_match(
        train, predictions, learned_rules, metrics, params
    )
    diagnostics, summary = diagnose_missed_triples(
        train,
        test,
        predictions,
        learned_rules,
        min_support=int(params["min_support"]),
        min_confidence=float(params["min_confidence"]),
        max_rules=int(params["max_rules"]),
        max_paths_per_pivot=int(params["max_paths_per_pivot"]),
        artifacts_match=artifacts_match,
        artifact_mismatch_reasons=mismatch_reasons,
    )
    summary["saved_artifacts_consistent"] = int(artifacts_match)
    return diagnostics, summary


def write_diagnostics(
    root: Path, diagnostics: list[dict[str, object]]
) -> Path:
    output_dir = root.resolve() / "results" / "diagnostics"
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output_path = output_dir / f"test_miss_diagnostics_{timestamp}.csv"
    columns = [
        "head",
        "relation",
        "tail",
        "diagnostic_category",
        "two_hop_path_exists",
        "matching_path_count",
        "path_examples_json",
        "learned_relevant_rule_exists",
        "relevant_rules_json",
        "excluded_matching_path_count",
        "excluded_applicable_path_count",
        "excluded_pivots_json",
        "path_limit",
        "path_examples_truncated",
        "evidence",
    ]
    with output_path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(diagnostics)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Diagnose missed test triples from training-path evidence."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Project root containing dataset and results directories.",
    )
    args = parser.parse_args()
    root = args.root.resolve()
    source_files = (
        Path(__file__),
        Path(__file__).with_name("kg_pipeline.py"),
        Path(__file__).with_name("run_provenance.py"),
    )
    run_snapshot = capture_run_snapshot(root, source_files)
    diagnostics, summary = analyze_project(args.root)
    output_path = write_diagnostics(args.root, diagnostics)
    saved_metrics = json.loads(
        (root / "results" / "metrics.json").read_text(encoding="utf-8")
    )
    run_timestamp = output_path.stem.removeprefix("test_miss_diagnostics_")
    categories = Counter(
        str(row["diagnostic_category"]) for row in diagnostics
    )
    write_run_manifest(
        root,
        output_path.parent,
        experiment_type="test_miss_diagnostics",
        run_timestamp_utc=run_timestamp,
        rule_settings=saved_metrics.get("parameters"),
        candidate_generation={
            "path_limit": saved_metrics.get("parameters", {}).get(
                "max_paths_per_pivot"
            ),
            "basis": (
                "Analyze saved prediction misses and inspect two-hop paths "
                "in the training graph."
            ),
        },
        scoring={"method": "not applicable; diagnostic classification only"},
        evaluation_definition={
            "diagnostic_scope": (
                "One row per missed test triple; categories are evidence "
                "classifications, not truth labels for unmatched candidates."
            ),
        },
        metrics={
            "missed_test_triples": len(diagnostics),
            "diagnostic_categories": dict(sorted(categories.items())),
            "saved_artifacts_consistent": summary["saved_artifacts_consistent"],
        },
        used_for_learning=("train",),
        used_for_evaluation=("test",),
        output_paths=(output_path,),
        manifest_filename=f"{output_path.stem}.manifest.json",
        snapshot=run_snapshot,
    )
    print(
        json.dumps(
            {"output": str(output_path), "summary": summary},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
