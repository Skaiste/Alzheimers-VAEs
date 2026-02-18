"""Index helpers for experiments."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from .schemas import IndexEntry
from .storage import append_jsonl_locked, read_jsonl, write_jsonl_atomic


def read_index(index_path: Path) -> list[IndexEntry]:
    return read_jsonl(index_path)


def append_index_entry(index_path: Path, entry: IndexEntry) -> None:
    append_jsonl_locked(index_path, entry)


def write_index(index_path: Path, entries: list[IndexEntry]) -> None:
    write_jsonl_atomic(index_path, entries)


def find_index_entry(index_path: Path, experiment_id: str) -> Optional[IndexEntry]:
    for entry in read_index(index_path):
        if entry.get("experiment_id") == experiment_id:
            return entry
    return None


def build_index_entry(metadata: dict, metadata_path: Path, base_dir: Path) -> IndexEntry:
    summary = metadata.get("summary", {})
    training_params = metadata.get("training_params", {})
    model_params = metadata.get("model_params", {})

    return {
        "schema_version": int(metadata.get("schema_version", 1)),
        "experiment_id": metadata["experiment_id"],
        "created_at": metadata["created_at"],
        "status": metadata.get("status", "unknown"),
        "model_type": metadata.get("model_type", "unknown"),
        "best_val_loss": _to_float(summary.get("best_val_loss")),
        "num_epochs": _to_int(summary.get("num_epochs")),
        "learning_rate": _to_float(training_params.get("learning_rate")),
        "latent_dim": _to_int(model_params.get("latent_dim")),
        "tags": list(metadata.get("tags", [])),
        "metadata_path": os.path.relpath(metadata_path, base_dir),
    }


def _to_float(value: object) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _to_int(value: object) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
