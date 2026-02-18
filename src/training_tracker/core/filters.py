"""Filtering helpers for experiment index rows."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Iterable


def apply_filters(rows: Iterable[dict], filters: dict | None) -> list[dict]:
    if not filters:
        return list(rows)

    output: list[dict] = []
    for row in rows:
        if _matches(row, filters):
            output.append(row)
    return output


def _matches(row: dict, filters: dict) -> bool:
    model_type = filters.get("model_type")
    if model_type and row.get("model_type") != model_type:
        return False

    status = filters.get("status")
    if status and row.get("status") != status:
        return False

    tags_contains = filters.get("tags_contains")
    if tags_contains:
        row_tags = set(row.get("tags", []))
        wanted = {tags_contains} if isinstance(tags_contains, str) else set(tags_contains)
        if not wanted.issubset(row_tags):
            return False

    if not _range_match(row, "best_val_loss", filters, "best_val_loss_min", "best_val_loss_max"):
        return False
    if not _range_match(row, "num_epochs", filters, "num_epochs_min", "num_epochs_max"):
        return False
    if not _range_match(row, "learning_rate", filters, "learning_rate_min", "learning_rate_max"):
        return False
    if not _range_match(row, "latent_dim", filters, "latent_dim_min", "latent_dim_max"):
        return False

    created_after = filters.get("created_after")
    if created_after and _parse_dt(row.get("created_at")) < _parse_dt(created_after):
        return False

    created_before = filters.get("created_before")
    if created_before and _parse_dt(row.get("created_at")) > _parse_dt(created_before):
        return False

    return True


def _range_match(row: dict, key: str, filters: dict, min_key: str, max_key: str) -> bool:
    value = row.get(key)
    if value is None:
        return filters.get(min_key) is None and filters.get(max_key) is None

    numeric = _to_float(value)
    if numeric is None:
        return False

    min_v = filters.get(min_key)
    if min_v is not None and numeric < float(min_v):
        return False

    max_v = filters.get(max_key)
    if max_v is not None and numeric > float(max_v):
        return False

    return True


def _to_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _parse_dt(value: Any) -> datetime:
    if not isinstance(value, str):
        return datetime.min
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return datetime.min
