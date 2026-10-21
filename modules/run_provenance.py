"""Create and verify provenance manifests for experiment outputs."""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path


MANIFEST_NAME = "run_manifest.json"
DATASET_SPLITS = ("train", "valid", "test")
REQUIRED_FIELDS = {
    "schema_version",
    "experiment_type",
    "run_timestamp_utc",
    "manifest_created_at_utc",
    "provenance_kind",
    "datasets",
    "code_version",
    "configured",
    "measured",
    "outputs",
    "external_outputs",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_version(root: Path, *, observed_at: str) -> dict[str, object]:
    commit_result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        capture_output=True,
        check=False,
        text=True,
    )
    status_result = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=root,
        capture_output=True,
        check=False,
        text=True,
    )
    return {
        "git_commit": (
            commit_result.stdout.strip()
            if commit_result.returncode == 0
            else None
        ),
        "working_tree_dirty": (
            bool(status_result.stdout.strip())
            if status_result.returncode == 0
            else None
        ),
        "observation_timing": observed_at,
    }


def capture_run_snapshot(
    root: Path, source_files: tuple[Path, ...]
) -> dict[str, object]:
    """Capture split and code identities before experiment work starts."""
    root = root.resolve()
    splits: dict[str, dict[str, object]] = {}
    for split in DATASET_SPLITS:
        path = root / "dataset" / f"{split}.txt"
        splits[split] = {
            "path": path.relative_to(root).as_posix(),
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
            "used_for_learning": False,
            "used_for_evaluation": False,
            "identity_timing": "hashed_at_run_start",
        }

    source_root = Path(__file__).resolve().parents[1]
    sources = []
    for path in source_files:
        resolved = path.resolve()
        try:
            source_name = resolved.relative_to(source_root).as_posix()
        except ValueError:
            source_name = resolved.as_uri()
        sources.append({"path": source_name, "sha256": sha256_file(resolved)})

    code_version = _git_version(root, observed_at="run_start")
    code_version.update(
        {
            "execution_version_verified": True,
            "source_path_base": source_root.as_posix(),
            "source_files_hashed_at_execution": sources,
            "historical_execution_version_note": None,
        }
    )
    return {
        "datasets": {
            "identity_timing": "at_run_start",
            "splits": splits,
        },
        "code_version": code_version,
    }


def validate_manifest(manifest: object) -> None:
    if not isinstance(manifest, dict):
        raise ValueError("manifest must be a JSON object")
    missing = REQUIRED_FIELDS - manifest.keys()
    if missing:
        raise ValueError(f"manifest is missing fields: {sorted(missing)}")
    if manifest["schema_version"] != 1:
        raise ValueError("unsupported manifest schema_version")
    if not isinstance(manifest["datasets"], dict) or not isinstance(
        manifest["datasets"].get("splits"), dict
    ):
        raise ValueError("manifest datasets.splits must be an object")
    for split in DATASET_SPLITS:
        record = manifest["datasets"]["splits"].get(split)
        if not isinstance(record, dict) or not {
            "path",
            "sha256",
            "size_bytes",
            "used_for_learning",
            "used_for_evaluation",
            "identity_timing",
        }.issubset(record):
            raise ValueError(f"manifest is missing dataset split metadata: {split}")
    code = manifest["code_version"]
    if not isinstance(code, dict):
        raise ValueError("manifest code_version must be an object")
    has_git_version = (
        {"git_commit", "working_tree_dirty", "observation_timing"}.issubset(code)
        or {
            "git_commit_observed_at_manifest_creation",
            "working_tree_dirty_at_manifest_creation",
        }.issubset(code)
    )
    if not has_git_version or not {
        "execution_version_verified",
        "source_files_hashed_at_execution",
    }.issubset(code):
        raise ValueError("manifest code_version metadata is incomplete")
    configured = manifest["configured"]
    measured = manifest["measured"]
    if not isinstance(configured, dict) or not {
        "rule_settings",
        "candidate_generation",
        "scoring",
    }.issubset(configured):
        raise ValueError("manifest configured settings are incomplete")
    if not isinstance(measured, dict) or not {
        "metrics",
        "evaluation_definition",
    }.issubset(measured):
        raise ValueError("manifest measured results are incomplete")
    if not isinstance(manifest["outputs"], list):
        raise ValueError("manifest outputs must be a list")
    if not isinstance(manifest["external_outputs"], list):
        raise ValueError("manifest external_outputs must be a list")
    for output in (*manifest["outputs"], *manifest["external_outputs"]):
        if not isinstance(output, dict) or not {
            "path",
            "sha256",
            "size_bytes",
        }.issubset(output):
            raise ValueError("manifest output metadata is incomplete")
    for output in manifest["external_outputs"]:
        if output.get("storage_status") != "excluded_from_git_commit":
            raise ValueError("external output must declare its storage status")
        if not output.get("reason"):
            raise ValueError("external output must explain why it is external")


def write_run_manifest(
    root: Path,
    experiment_dir: Path,
    *,
    experiment_type: str,
    run_timestamp_utc: str,
    rule_settings: dict[str, object] | None,
    candidate_generation: dict[str, object],
    scoring: dict[str, object],
    evaluation_definition: dict[str, object],
    metrics: dict[str, object],
    used_for_learning: tuple[str, ...],
    used_for_evaluation: tuple[str, ...],
    provenance_kind: str = "captured_at_run",
    output_paths: tuple[Path, ...] | None = None,
    external_output_paths: tuple[Path, ...] = (),
    manifest_filename: str = MANIFEST_NAME,
    snapshot: dict[str, object] | None = None,
) -> Path:
    root = root.resolve()
    experiment_dir = experiment_dir.resolve()
    if provenance_kind not in {"captured_at_run", "retrospective_from_saved_outputs"}:
        raise ValueError(f"unsupported provenance kind: {provenance_kind}")

    source_root = Path(__file__).resolve().parents[1]
    if provenance_kind == "captured_at_run":
        if snapshot is None:
            raise ValueError("captured_at_run manifests require a run-start snapshot")
        datasets = snapshot["datasets"]
        code_version = snapshot["code_version"]
        for split in DATASET_SPLITS:
            datasets["splits"][split]["used_for_learning"] = (
                split in used_for_learning
            )
            datasets["splits"][split]["used_for_evaluation"] = (
                split in used_for_evaluation
            )
    else:
        if snapshot is not None:
            raise ValueError("retrospective manifests cannot use a run-start snapshot")
        split_records: dict[str, dict[str, object]] = {}
        for split in DATASET_SPLITS:
            path = root / "dataset" / f"{split}.txt"
            split_records[split] = {
                "path": path.relative_to(root).as_posix(),
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
                "used_for_learning": split in used_for_learning,
                "used_for_evaluation": split in used_for_evaluation,
                "identity_timing": "hashed_at_manifest_creation",
            }
        code_version = _git_version(root, observed_at="manifest_creation")
        code_version.update(
            {
                "execution_version_verified": False,
                "source_path_base": source_root.as_posix(),
                "source_files_hashed_at_execution": [],
                "historical_execution_version_note": (
                    "Execution-time commit, worktree state, and source hashes "
                    "were not recorded; the current commit is not asserted as "
                    "the version used by this historical run."
                ),
            }
        )
        datasets = {
            "identity_timing": "at_manifest_creation",
            "splits": split_records,
        }

    external_paths = {path.resolve() for path in external_output_paths}
    files = (
        sorted(experiment_dir.rglob("*"))
        if output_paths is None
        else sorted(output_paths)
    )
    outputs = []
    for path in files:
        if (
            not path.is_file()
            or path.name == manifest_filename
            or path.resolve() in external_paths
        ):
            continue
        outputs.append(
            {
                "path": path.resolve().relative_to(root).as_posix(),
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
        )
    external_outputs = []
    for path in sorted(external_paths):
        if not path.is_file():
            raise FileNotFoundError(path)
        external_outputs.append(
            {
                "path": path.relative_to(root).as_posix(),
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
                "storage_status": "excluded_from_git_commit",
                "reason": (
                    "Large SQLite experiment output intentionally retained "
                    "outside this source commit."
                ),
            }
        )

    manifest: dict[str, object] = {
        "schema_version": 1,
        "experiment_type": experiment_type,
        "run_timestamp_utc": run_timestamp_utc,
        "manifest_created_at_utc": datetime.now(timezone.utc).isoformat(),
        "provenance_kind": provenance_kind,
        "datasets": datasets,
        "code_version": code_version,
        "configured": {
            "rule_settings": rule_settings,
            "candidate_generation": candidate_generation,
            "scoring": scoring,
        },
        "measured": {
            "metrics": metrics,
            "evaluation_definition": evaluation_definition,
        },
        "outputs": outputs,
        "external_outputs": external_outputs,
    }
    validate_manifest(manifest)
    manifest_path = experiment_dir / manifest_filename
    with manifest_path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(manifest, stream, indent=2)
        stream.write("\n")
    return manifest_path


def verify_output_hashes(root: Path, manifest: object) -> list[str]:
    validate_manifest(manifest)
    errors: list[str] = []
    for output in (*manifest["outputs"], *manifest["external_outputs"]):
        path = root.resolve() / output["path"]
        if not path.is_file():
            errors.append(f"missing output: {output['path']}")
            continue
        actual = sha256_file(path)
        if actual != output["sha256"]:
            errors.append(f"hash mismatch: {output['path']}")
        if path.stat().st_size != output["size_bytes"]:
            errors.append(f"size mismatch: {output['path']}")
    return errors
