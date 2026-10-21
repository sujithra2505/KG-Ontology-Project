"""Compare baseline and distinct-rule combined scores on validation only."""

from __future__ import annotations

import argparse
import itertools
import json
import math
import sqlite3
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

if __package__:
    from .kg_cap_experiments import (
        _PeakMemorySampler,
        _rss_bytes,
        _validation_metrics,
        _write_json_exclusive,
    )
    from .kg_pipeline import PathRule, RuleBody, Triple, learn_path_rules, read_triples
    from .run_provenance import capture_run_snapshot, write_run_manifest
else:
    from kg_cap_experiments import (
        _PeakMemorySampler,
        _rss_bytes,
        _validation_metrics,
        _write_json_exclusive,
    )
    from kg_pipeline import PathRule, RuleBody, Triple, learn_path_rules, read_triples
    from run_provenance import capture_run_snapshot, write_run_manifest


MIN_SUPPORT = 2
MIN_CONFIDENCE = 0.2
MAX_RULES = 3_000
MAX_PATHS_PER_PIVOT = 10_000
INSERT_BATCH_SIZE = 2_000
EVIDENCE_BATCH_SIZE = 5_000
COMMIT_BATCH_SIZE = 50_000


def combine_distinct_rule_confidences(
    evidence: Iterable[tuple[tuple[str, str, str], float]],
) -> float:
    """Noisy-OR over unique body/target rules; duplicate paths count once."""
    confidence_by_rule: dict[tuple[str, str, str], float] = {}
    for rule_key, confidence in evidence:
        if not 0 <= confidence <= 1:
            raise ValueError("rule confidence must be between 0 and 1")
        confidence_by_rule[rule_key] = max(
            confidence_by_rule.get(rule_key, 0.0), confidence
        )
    if any(confidence == 1 for confidence in confidence_by_rule.values()):
        return 1.0
    return 1 - math.prod(
        1 - confidence for confidence in confidence_by_rule.values()
    )


def _insert_batch(
    connection: sqlite3.Connection,
    candidates: list[tuple[str, str, str, float, int]],
    evidence: list[tuple[str, str, str, str, str, float, int]],
) -> None:
    if candidates:
        connection.executemany(
            "INSERT INTO candidates (head, relation, tail, baseline_score, "
            "baseline_support) VALUES (?, ?, ?, ?, ?) "
            "ON CONFLICT(head, relation, tail) DO UPDATE SET "
            "baseline_score = MAX(baseline_score, excluded.baseline_score), "
            "baseline_support = MAX(baseline_support, excluded.baseline_support)",
            candidates,
        )
        candidates.clear()
    if evidence:
        connection.executemany(
            "INSERT OR IGNORE INTO rule_evidence "
            "(head, relation, tail, body1, body2, confidence, support) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            evidence,
        )
        evidence.clear()


def generate_scoring_evidence(
    train: list[Triple],
    rules: Iterable[PathRule],
    max_paths_per_pivot: int,
    database_path: Path,
) -> dict[str, int]:
    """Stream candidates and deduplicated per-rule evidence into SQLite."""
    if database_path.exists():
        raise FileExistsError(database_path)
    outgoing: dict[str, list[tuple[str, str]]] = defaultdict(list)
    incoming: dict[str, list[tuple[str, str]]] = defaultdict(list)
    known = set(train)
    for head, relation, tail in train:
        outgoing[head].append((relation, tail))
        incoming[tail].append((relation, head))

    rules_by_body: dict[RuleBody, list[PathRule]] = defaultdict(list)
    for rule in rules:
        rules_by_body[(rule.first_relation, rule.second_relation)].append(rule)

    connection = sqlite3.connect(database_path)
    try:
        connection.execute("PRAGMA cache_size = -4096")
        connection.execute("PRAGMA temp_store = FILE")
        connection.execute(
            "CREATE TABLE candidates (head TEXT, relation TEXT, tail TEXT, "
            "baseline_score REAL NOT NULL, baseline_support INTEGER NOT NULL, "
            "PRIMARY KEY (head, relation, tail)) WITHOUT ROWID"
        )
        connection.execute(
            "CREATE TABLE rule_evidence (head TEXT, relation TEXT, tail TEXT, "
            "body1 TEXT, body2 TEXT, confidence REAL NOT NULL, support INTEGER "
            "NOT NULL, PRIMARY KEY (head, relation, tail, body1, body2)) "
            "WITHOUT ROWID"
        )
        connection.execute(
            "CREATE TABLE validation (head TEXT, relation TEXT, tail TEXT, "
            "PRIMARY KEY (head, relation, tail)) WITHOUT ROWID"
        )
        connection.execute(
            "CREATE TABLE combined_scores (head TEXT, relation TEXT, tail TEXT, "
            "score REAL NOT NULL, PRIMARY KEY (head, relation, tail)) WITHOUT ROWID"
        )
        connection.commit()

        skipped_pivots = 0
        skipped_paths = 0
        candidate_batch: list[tuple[str, str, str, float, int]] = []
        evidence_batch: list[tuple[str, str, str, str, str, float, int]] = []
        pending_commit = 0
        for pivot, incoming_edges in incoming.items():
            outgoing_edges = outgoing.get(pivot, ())
            possible_paths = len(incoming_edges) * len(outgoing_edges)
            if possible_paths > max_paths_per_pivot:
                skipped_pivots += 1
                skipped_paths += possible_paths
                continue
            for first_relation, head in incoming_edges:
                for second_relation, tail in outgoing_edges:
                    for rule in rules_by_body.get(
                        (first_relation, second_relation), ()
                    ):
                        triple = (head, rule.predicted_relation, tail)
                        if triple in known:
                            continue
                        candidate_batch.append(
                            (*triple, rule.confidence, rule.support)
                        )
                        evidence_batch.append(
                            (
                                *triple,
                                rule.first_relation,
                                rule.second_relation,
                                rule.confidence,
                                rule.support,
                            )
                        )
                        if len(candidate_batch) >= INSERT_BATCH_SIZE:
                            _insert_batch(
                                connection, candidate_batch, evidence_batch
                            )
                            pending_commit += INSERT_BATCH_SIZE
                            if pending_commit >= COMMIT_BATCH_SIZE:
                                connection.commit()
                                pending_commit = 0
            _insert_batch(connection, candidate_batch, evidence_batch)

        _insert_batch(connection, candidate_batch, evidence_batch)
        connection.commit()
        return {
            "skipped_high_degree_pivots": skipped_pivots,
            "skipped_two_hop_paths": skipped_paths,
            "unique_candidate_edges": connection.execute(
                "SELECT COUNT(*) FROM candidates"
            ).fetchone()[0],
            "unique_candidate_rule_evidence": connection.execute(
                "SELECT COUNT(*) FROM rule_evidence"
            ).fetchone()[0],
        }
    finally:
        connection.close()


def _compute_combined_scores(database_path: Path) -> dict[str, int | float]:
    connection = sqlite3.connect(database_path)
    try:
        cursor = connection.execute(
            "SELECT e.head, e.relation, e.tail, e.confidence, c.baseline_score "
            "FROM rule_evidence e JOIN candidates c USING (head, relation, tail) "
            "ORDER BY e.head, e.relation, e.tail, e.body1, e.body2"
        )
        insert_batch: list[tuple[str, str, str, float]] = []
        score_delta_sum = 0.0
        changed_candidate_count = 0
        for key, grouped_rows in itertools.groupby(
            cursor, key=lambda row: (row[0], row[1], row[2])
        ):
            first = next(grouped_rows)
            baseline_score = float(first[4])
            confidences = [float(first[3])]
            confidences.extend(float(row[3]) for row in grouped_rows)
            combined_score = combine_distinct_rule_confidences(
                [
                    ((str(index), "body", "target"), confidence)
                    for index, confidence in enumerate(confidences)
                ]
            )
            score_delta_sum += combined_score - baseline_score
            changed_candidate_count += int(
                abs(combined_score - baseline_score) > 1e-12
            )
            insert_batch.append((*key, combined_score))
            if len(insert_batch) >= EVIDENCE_BATCH_SIZE:
                connection.executemany(
                    "INSERT INTO combined_scores (head, relation, tail, score) "
                    "VALUES (?, ?, ?, ?)",
                    insert_batch,
                )
                connection.commit()
                insert_batch.clear()
        if insert_batch:
            connection.executemany(
                "INSERT INTO combined_scores (head, relation, tail, score) "
                "VALUES (?, ?, ?, ?)",
                insert_batch,
            )
            connection.commit()
        candidate_count = connection.execute(
            "SELECT COUNT(*) FROM candidates"
        ).fetchone()[0]
        return {
            "unique_candidates_with_score": connection.execute(
                "SELECT COUNT(*) FROM combined_scores"
            ).fetchone()[0],
            "candidates_with_changed_score": changed_candidate_count,
            "mean_score_delta_from_baseline": (
                score_delta_sum / candidate_count if candidate_count else 0.0
            ),
        }
    finally:
        connection.close()


def _score_statistics(
    database_path: Path,
) -> dict[str, dict[str, int | float]]:
    connection = sqlite3.connect(database_path)
    try:
        baseline = connection.execute(
            "SELECT MIN(baseline_score), AVG(baseline_score), "
            "MAX(baseline_score) FROM candidates"
        ).fetchone()
        combined = connection.execute(
            "SELECT MIN(score), AVG(score), MAX(score) FROM combined_scores"
        ).fetchone()
        multi_rule_candidates = connection.execute(
            "SELECT COUNT(*) FROM (SELECT head, relation, tail "
            "FROM rule_evidence GROUP BY head, relation, tail HAVING COUNT(*) > 1)"
        ).fetchone()[0]
    finally:
        connection.close()
    return {
        "baseline_max_rule_confidence": {
            "minimum": baseline[0] or 0.0,
            "mean": baseline[1] or 0.0,
            "maximum": baseline[2] or 0.0,
        },
        "distinct_rule_noisy_or": {
            "minimum": combined[0] or 0.0,
            "mean": combined[1] or 0.0,
            "maximum": combined[2] or 0.0,
        },
        "candidates_with_multiple_distinct_rules": multi_rule_candidates,
    }


def run_scoring_experiment(root: Path) -> Path:
    root = root.resolve()
    source_files = (
        Path(__file__),
        Path(__file__).with_name("kg_cap_experiments.py"),
        Path(__file__).with_name("kg_pipeline.py"),
        Path(__file__).with_name("run_provenance.py"),
    )
    run_snapshot = capture_run_snapshot(root, source_files)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output_dir = root / "results" / "experiments" / f"candidate_scoring_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=False)
    database_path = output_dir / "scoring.sqlite"

    train = read_triples(root / "dataset" / "train.txt")
    validation = read_triples(root / "dataset" / "valid.txt")
    baseline_rss = _rss_bytes()
    started = time.perf_counter()
    with _PeakMemorySampler() as memory:
        rules, mining_stats = learn_path_rules(
            train,
            min_support=MIN_SUPPORT,
            min_confidence=MIN_CONFIDENCE,
            max_rules=MAX_RULES,
            max_paths_per_pivot=MAX_PATHS_PER_PIVOT,
        )
        prediction_stats = generate_scoring_evidence(
            train, rules, MAX_PATHS_PER_PIVOT, database_path
        )
        combined_stats = _compute_combined_scores(database_path)
        baseline_metrics = _validation_metrics(
            database_path,
            validation,
            prediction_stats["unique_candidate_edges"],
        )
        combined_metrics = _validation_metrics(
            database_path,
            validation,
            prediction_stats["unique_candidate_edges"],
            candidate_table="combined_scores",
        )
        peak_rss = memory.peak_rss_bytes
    elapsed = time.perf_counter() - started
    score_stats = _score_statistics(database_path)

    result = {
        "run_timestamp_utc": timestamp,
        "dataset_files_used": ["dataset/train.txt", "dataset/valid.txt"],
        "test_split_used": False,
        "parameters": {
            "min_support": MIN_SUPPORT,
            "min_confidence": MIN_CONFIDENCE,
            "max_rules": MAX_RULES,
            "max_paths_per_pivot": MAX_PATHS_PER_PIVOT,
        },
        "learned_rules": len(rules),
        "mining": mining_stats,
        "prediction": prediction_stats,
        "scoring_methods": {
            "baseline": {
                "name": "maximum supporting-rule confidence",
                "description": (
                    "Maximum confidence of any learned rule supporting a "
                    "candidate; preserves the current baseline scoring."
                ),
                "validation": baseline_metrics,
            },
            "experimental": {
                "name": "noisy-OR over distinct supporting rules",
                "description": (
                    "For each candidate, combine one confidence per unique "
                    "(body relation 1, body relation 2, target relation) rule "
                    "as 1 - product(1 - confidence). Repeated paths for the "
                    "same rule are stored once."
                ),
                "assumption": (
                    "Distinct rule evidence is treated as independent for "
                    "scoring; related rules can violate this assumption."
                ),
                "validation": combined_metrics,
            },
        },
        "score_statistics": score_stats,
        "score_comparison": combined_stats,
        "candidate_coverage_unchanged": (
            baseline_metrics["known_edges"] == combined_metrics["known_edges"]
            and baseline_metrics["predicted_edges_found"]
            == combined_metrics["predicted_edges_found"]
            and baseline_metrics["precision"] == combined_metrics["precision"]
            and baseline_metrics["recall"] == combined_metrics["recall"]
        ),
        "ranked_metric_status": (
            "No ranking improvement is claimed. The current candidate universe "
            "is an incomplete, rule-generated subset and is not a defined "
            "filtered full-ranking protocol."
        ),
        "recommended_future_ranking_protocol": {
            "candidate_universe": (
                "For each validation query, rank all entity replacements for "
                "head and tail from the training entity universe, evaluating "
                "head and tail queries separately and averaged."
            ),
            "filtered_evaluation": (
                "Filter other known true completions from train, validation, "
                "and test, then restore the target triple. Test labels must "
                "not be accessed during tuning."
            ),
            "tie_handling": (
                "Use average rank for equal scores; compute reciprocal rank "
                "from that rank and Hits@K from rank <= K."
            ),
            "sampling": (
                "Prefer full entity ranking; if resource limits require "
                "sampling, record candidate count, seed, and sampling method "
                "and label the result sampled."
            ),
        },
        "runtime_seconds": elapsed,
        "memory": {
            "baseline_rss_bytes": baseline_rss,
            "peak_sampled_rss_bytes": peak_rss,
            "peak_sampled_rss_delta_bytes": max(0, peak_rss - baseline_rss),
            "sampling_interval_seconds": 0.05,
        },
        "candidate_database": database_path.name,
        "candidate_database_bytes": database_path.stat().st_size,
        "commands": [
            "py -3.13 modules\\kg_scoring_experiment.py",
        ],
    }
    _write_json_exclusive(output_dir / "metrics.json", result)
    write_run_manifest(
        root,
        output_dir,
        experiment_type="candidate_scoring_comparison",
        run_timestamp_utc=timestamp,
        rule_settings={
            "min_support": MIN_SUPPORT,
            "min_confidence": MIN_CONFIDENCE,
            "max_rules": MAX_RULES,
            "max_paths_per_pivot": MAX_PATHS_PER_PIVOT,
        },
        candidate_generation={
            "max_paths_per_pivot": MAX_PATHS_PER_PIVOT,
            "candidate_membership_rule": (
                "Generate candidates from eligible training two-hop paths and "
                "learned rules; exclude training triples."
            ),
            "duplicate_candidates": "Deduplicated in SQLite.",
        },
        scoring={
            "baseline": "Maximum supporting-rule confidence.",
            "experimental": (
                "Noisy-OR across unique supporting rule body/target triples; "
                "repeated paths for the same rule count once."
            ),
            "assumption": (
                "Distinct rule evidence is treated as independent; correlated "
                "rules can violate this assumption."
            ),
        },
        evaluation_definition={
            "precision": (
                "Unique candidate/validation exact triple matches divided by "
                "unique generated candidates."
            ),
            "recall": (
                "Unique candidate/validation exact triple matches divided by "
                "unique known validation triples."
            ),
            "ranking_metrics": (
                "Not reported; the candidate universe is a partial "
                "rule-generated set without a filtered ranking protocol."
            ),
            "unmatched_candidates": "Unknown truth; not treated as false.",
        },
        metrics={
            "learned_rules": len(rules),
            "prediction": prediction_stats,
            "scoring_methods": result["scoring_methods"],
            "score_comparison": combined_stats,
            "runtime_seconds": elapsed,
            "memory": result["memory"],
        },
        used_for_learning=("train",),
        used_for_evaluation=("valid",),
        snapshot=run_snapshot,
    )
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare candidate scoring methods using validation only."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Project root containing dataset and results directories.",
    )
    args = parser.parse_args()
    output_dir = run_scoring_experiment(args.root)
    print(json.dumps({"output_dir": str(output_dir)}, indent=2))


if __name__ == "__main__":
    main()
