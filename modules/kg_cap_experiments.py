"""Run memory-bounded path-cap experiments using training and validation only."""

from __future__ import annotations

import argparse
import ctypes
import json
import sqlite3
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable
from ctypes import wintypes

if __package__:
    from .kg_pipeline import (
        PathRule,
        RuleBody,
        Triple,
        learn_path_rules,
        read_triples,
    )
    from .run_provenance import capture_run_snapshot, write_run_manifest
else:
    from kg_pipeline import (
        PathRule,
        RuleBody,
        Triple,
        learn_path_rules,
        read_triples,
    )
    from run_provenance import capture_run_snapshot, write_run_manifest


DEFAULT_CAPS = (2_000, 5_000, 10_000)
INSERT_BATCH_SIZE = 2_000
COMMIT_BATCH_SIZE = 50_000
DEFAULT_MIN_SUPPORT = 2
DEFAULT_MIN_CONFIDENCE = 0.1
DEFAULT_MAX_RULES = 3_000


def _write_json_exclusive(path: Path, value: object) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")


def _insert_candidates(
    connection: sqlite3.Connection,
    batch: list[Triple],
) -> None:
    if batch:
        connection.executemany(
            "INSERT OR IGNORE INTO candidates (head, relation, tail) "
            "VALUES (?, ?, ?)",
            batch,
        )
        batch.clear()


def _rss_bytes() -> int:
    class ProcessMemoryCounters(ctypes.Structure):
        _fields_ = [
            ("cb", ctypes.c_ulong),
            ("PageFaultCount", ctypes.c_ulong),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
        ]

    counters = ProcessMemoryCounters()
    counters.cb = ctypes.sizeof(counters)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    get_process_memory_info = psapi.GetProcessMemoryInfo
    get_process_memory_info.argtypes = [
        wintypes.HANDLE,
        ctypes.POINTER(ProcessMemoryCounters),
        wintypes.DWORD,
    ]
    get_process_memory_info.restype = wintypes.BOOL
    succeeded = get_process_memory_info(
        kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb
    )
    if not succeeded:
        raise ctypes.WinError(ctypes.get_last_error())
    return int(counters.WorkingSetSize)


def stream_candidates_to_sqlite(
    train: list[Triple],
    rules: Iterable[PathRule],
    max_paths_per_pivot: int,
    database_path: Path,
) -> dict[str, int]:
    """Generate and deduplicate candidates on disk, without a candidate dict."""
    if database_path.exists():
        raise FileExistsError(database_path)
    outgoing: dict[str, list[tuple[str, str]]] = {}
    incoming: dict[str, list[tuple[str, str]]] = {}
    known = set(train)
    for head, relation, tail in train:
        outgoing.setdefault(head, []).append((relation, tail))
        incoming.setdefault(tail, []).append((relation, head))

    rules_by_body: dict[RuleBody, list[PathRule]] = {}
    for rule in rules:
        rules_by_body.setdefault(
            (rule.first_relation, rule.second_relation), []
        ).append(rule)

    connection = sqlite3.connect(database_path)
    try:
        connection.execute("PRAGMA cache_size = -4096")
        connection.execute("PRAGMA temp_store = FILE")
        connection.execute("CREATE TABLE candidates (head TEXT, relation TEXT, tail TEXT, PRIMARY KEY (head, relation, tail)) WITHOUT ROWID")
        connection.execute(
            "CREATE TABLE validation (head TEXT, relation TEXT, tail TEXT, "
            "PRIMARY KEY (head, relation, tail)) WITHOUT ROWID"
        )
        connection.commit()

        skipped_pivots = 0
        skipped_paths = 0
        batch: list[Triple] = []
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
                        candidate = (head, rule.predicted_relation, tail)
                        if candidate in known:
                            continue
                        batch.append(candidate)
                        if len(batch) >= INSERT_BATCH_SIZE:
                            _insert_candidates(connection, batch)
                            pending_commit += INSERT_BATCH_SIZE
                            if pending_commit >= COMMIT_BATCH_SIZE:
                                connection.commit()
                                pending_commit = 0

        _insert_candidates(connection, batch)
        connection.commit()
        return {
            "skipped_high_degree_pivots": skipped_pivots,
            "skipped_two_hop_paths": skipped_paths,
            "unique_candidate_edges": connection.execute(
                "SELECT COUNT(*) FROM candidates"
            ).fetchone()[0],
        }
    finally:
        connection.close()


def _validation_metrics(
    database_path: Path,
    validation: list[Triple],
    candidate_count: int,
    *,
    candidate_table: str = "candidates",
) -> dict[str, int | float]:
    if candidate_table not in {"candidates", "combined_scores"}:
        raise ValueError(f"Unsupported candidate table: {candidate_table}")
    connection = sqlite3.connect(database_path)
    try:
        connection.executemany(
            "INSERT OR IGNORE INTO validation (head, relation, tail) "
            "VALUES (?, ?, ?)",
            validation,
        )
        connection.commit()
        hits = connection.execute(
            f"SELECT COUNT(*) FROM {candidate_table} "
            "JOIN validation USING (head, relation, tail)"
        ).fetchone()[0]
    finally:
        connection.close()
    return {
        "known_edges": len(validation),
        "predicted_edges_found": hits,
        "precision": hits / candidate_count if candidate_count else 0.0,
        "recall": hits / len(validation) if validation else 0.0,
    }


class _PeakMemorySampler:
    def __init__(self, interval_seconds: float = 0.05) -> None:
        self._interval = interval_seconds
        self._stop = threading.Event()
        self.peak_rss_bytes = _rss_bytes()
        self._thread = threading.Thread(target=self._sample, daemon=True)

    def _sample(self) -> None:
        while not self._stop.wait(self._interval):
            self.peak_rss_bytes = max(self.peak_rss_bytes, _rss_bytes())

    def __enter__(self) -> _PeakMemorySampler:
        self._thread.start()
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self._stop.set()
        self._thread.join()
        self.peak_rss_bytes = max(self.peak_rss_bytes, _rss_bytes())


def run_cap_experiment(
    train: list[Triple],
    validation: list[Triple],
    *,
    max_paths_per_pivot: int,
    database_path: Path,
    min_support: int = DEFAULT_MIN_SUPPORT,
    min_confidence: float = DEFAULT_MIN_CONFIDENCE,
    max_rules: int = DEFAULT_MAX_RULES,
) -> dict[str, object]:
    if max_paths_per_pivot < 1:
        raise ValueError("max_paths_per_pivot must be positive")
    database_path.parent.mkdir(parents=True, exist_ok=True)
    if database_path.exists():
        raise FileExistsError(database_path)

    start_rss = _rss_bytes()
    overall_started = time.perf_counter()
    with _PeakMemorySampler() as memory:
        mining_started = time.perf_counter()
        rules, mining_stats = learn_path_rules(
            train,
            min_support=min_support,
            min_confidence=min_confidence,
            max_rules=max_rules,
            max_paths_per_pivot=max_paths_per_pivot,
        )
        mining_seconds = time.perf_counter() - mining_started

        prediction_started = time.perf_counter()
        candidate_stats = stream_candidates_to_sqlite(
            train, rules, max_paths_per_pivot, database_path
        )
        prediction_seconds = time.perf_counter() - prediction_started
        validation_metrics = _validation_metrics(
            database_path,
            list(set(validation)),
            candidate_stats["unique_candidate_edges"],
        )
        peak_rss_bytes = memory.peak_rss_bytes

    return {
        "parameters": {
            "min_support": min_support,
            "min_confidence": min_confidence,
            "max_rules": max_rules,
            "max_paths_per_pivot": max_paths_per_pivot,
        },
        "training_edges": len(set(train)),
        "validation_edges": len(set(validation)),
        "learned_rules": len(rules),
        "mining": mining_stats,
        "prediction": {
            "skipped_high_degree_pivots": candidate_stats[
                "skipped_high_degree_pivots"
            ],
            "skipped_two_hop_paths": candidate_stats["skipped_two_hop_paths"],
            "unique_candidate_edges": candidate_stats["unique_candidate_edges"],
        },
        "validation": validation_metrics,
        "runtime_seconds": {
            "rule_learning": mining_seconds,
            "candidate_generation_and_disk_deduplication": prediction_seconds,
            "total": time.perf_counter() - overall_started,
        },
        "memory": {
            "baseline_rss_bytes": start_rss,
            "peak_sampled_rss_bytes": peak_rss_bytes,
            "peak_sampled_rss_delta_bytes": max(0, peak_rss_bytes - start_rss),
            "sampling_interval_seconds": 0.05,
        },
        "candidate_database": database_path.name,
        "candidate_database_bytes": database_path.stat().st_size,
        "evaluation_definition": {
            "precision": "unique candidate/validation exact triple matches divided by unique generated candidates",
            "recall": "unique candidate/validation exact triple matches divided by known validation triples",
            "unmatched_candidates": "unknown truth; not treated as false",
        },
    }


def run_experiments(
    root: Path,
    *,
    caps: tuple[int, ...] = DEFAULT_CAPS,
) -> Path:
    root = root.resolve()
    source_files = (
        Path(__file__),
        Path(__file__).with_name("kg_pipeline.py"),
        Path(__file__).with_name("run_provenance.py"),
    )
    run_snapshot = capture_run_snapshot(root, source_files)
    train = read_triples(root / "dataset" / "train.txt")
    validation = read_triples(root / "dataset" / "valid.txt")
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output_dir = root / "results" / "experiments" / f"path_caps_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=False)

    results: list[dict[str, object]] = []
    for cap in caps:
        cap_dir = output_dir / f"cap_{cap}"
        cap_dir.mkdir(exist_ok=False)
        database_path = cap_dir / "candidates.sqlite"
        result = run_cap_experiment(
            train,
            validation,
            max_paths_per_pivot=cap,
            database_path=database_path,
        )
        result["configuration_id"] = cap_dir.name
        _write_json_exclusive(cap_dir / "metrics.json", result)
        results.append(result)

    summary = {
        "run_timestamp_utc": timestamp,
        "dataset_files_used": ["dataset/train.txt", "dataset/valid.txt"],
        "test_split_used": False,
        "shared_rule_parameters": {
            "min_support": DEFAULT_MIN_SUPPORT,
            "min_confidence": DEFAULT_MIN_CONFIDENCE,
            "max_rules": DEFAULT_MAX_RULES,
        },
        "caps": list(caps),
        "command": (
            "py -3.13 modules\\kg_cap_experiments.py --caps "
            + " ".join(str(cap) for cap in caps)
        ),
        "results": results,
    }
    _write_json_exclusive(output_dir / "summary.json", summary)
    write_run_manifest(
        root,
        output_dir,
        experiment_type="path_cap_comparison",
        run_timestamp_utc=timestamp,
        rule_settings=summary["shared_rule_parameters"],
        candidate_generation={
            "configured_path_caps": list(caps),
            "path_cap_semantics": (
                "Skip a pivot when incoming_degree * outgoing_degree exceeds "
                "the configured limit."
            ),
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
            "unmatched_candidates": "Unknown truth; not treated as false.",
        },
        metrics={"results": results},
        used_for_learning=("train",),
        used_for_evaluation=("valid",),
        snapshot=run_snapshot,
    )
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare pivot path caps using train and validation only."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Project root containing dataset and results directories.",
    )
    parser.add_argument(
        "--caps",
        nargs="+",
        type=int,
        default=list(DEFAULT_CAPS),
        help="Path limits to compare; all other rule parameters stay fixed.",
    )
    args = parser.parse_args()
    output_dir = run_experiments(args.root, caps=tuple(args.caps))
    print(json.dumps({"output_dir": str(output_dir)}, indent=2))


if __name__ == "__main__":
    main()
