from __future__ import annotations

import json
import re
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Any

from setup_index.feed_documents_upsert import feed_documents
from setup_index.file_utils import get_source_id


FACET_FIELDS = [
    "job_number",
    "doc_type",
    "ships",
    "ship_classes",
    "years_mentioned",
    "rates_mentioned",
    "shipyards_mentioned",
]


def _load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return data
    except Exception as e:
        print(f"Warning: failed to load {path}: {e}")
        return default


def _save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)


def load_facets(path: Path) -> dict[str, dict[str, int]]:
    data = _load_json(path, {})
    if not isinstance(data, dict):
        return {}
    cleaned: dict[str, dict[str, int]] = {}
    for field, mapping in data.items():
        if isinstance(mapping, dict):
            cleaned[field] = {}
            for key, value in mapping.items():
                try:
                    cleaned[field][str(key)] = int(value)
                except Exception:
                    pass
    return cleaned


def load_source_contributions(path: Path) -> dict[str, dict[str, list[str]]]:
    data = _load_json(path, {})
    if not isinstance(data, dict):
        return {}
    cleaned: dict[str, dict[str, list[str]]] = {}
    for source_id, field_map in data.items():
        if not isinstance(field_map, dict):
            continue
        cleaned[source_id] = {}
        for field, values in field_map.items():
            if isinstance(values, list):
                cleaned[source_id][field] = [str(v) for v in values]
    return cleaned


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def normalize_job_number(value: str) -> str:
    return normalize_whitespace(value).upper()


def normalize_doc_type(value: str) -> str:
    return normalize_whitespace(value).lower()


def normalize_ship(value: str) -> str:
    value = normalize_whitespace(value)
    value = value.replace("–", "-").replace("—", "-")
    value = re.sub(r"[’']s\b", "", value, flags=re.IGNORECASE)
    value = re.sub(r"\bdeck log\b", "", value, flags=re.IGNORECASE)
    value = re.sub(r"-?class\b", "", value, flags=re.IGNORECASE)
    value = normalize_whitespace(value)

    words = value.split()
    out = []
    for word in words:
        if word.upper() == "USS":
            out.append("USS")
        else:
            pieces = word.split("-")
            pieces = [p.capitalize() if p else p for p in pieces]
            out.append("-".join(pieces))

    value = " ".join(out)
    return normalize_whitespace(value)


def normalize_ship_class(value: str) -> str:
    value = normalize_whitespace(value)
    value = value.replace("–", "-").replace("—", "-")

    words = []
    for token in value.split():
        if token.upper() == "USS":
            words.append("USS")
        else:
            parts = token.split("-")
            parts = [p.capitalize() if p else p for p in parts]
            words.append("-".join(parts))

    value = " ".join(words)

    if not value.lower().endswith("-class"):
        value = re.sub(r"\bclass\b$", "", value, flags=re.IGNORECASE).strip()
        value = value + "-class"

    return normalize_whitespace(value)


def normalize_year(value: Any) -> str:
    return str(value).strip()


def normalize_rate(value: str) -> str:
    value = normalize_whitespace(value)
    words = []
    for token in value.split():
        pieces = token.split("-")
        pieces = [p.capitalize() if p else p for p in pieces]
        words.append("-".join(pieces))
    return " ".join(words)


def normalize_shipyard(value: str) -> str:
    value = normalize_whitespace(value)
    words = []
    for token in value.split():
        pieces = token.split("-")
        pieces = [p.capitalize() if p else p for p in pieces]
        words.append("-".join(pieces))
    return " ".join(words)


def normalize_facet_value(field: str, value: Any) -> str | None:
    if value is None:
        return None

    if not isinstance(value, str):
        if field == "years_mentioned":
            return normalize_year(value)
        value = str(value)

    value = value.strip()
    if not value:
        return None

    if field == "job_number":
        value = normalize_job_number(value)
    elif field == "doc_type":
        value = normalize_doc_type(value)
    elif field == "ships":
        value = normalize_ship(value)
    elif field == "ship_classes":
        value = normalize_ship_class(value)
    elif field == "years_mentioned":
        value = normalize_year(value)
    elif field == "rates_mentioned":
        value = normalize_rate(value)
    elif field == "shipyards_mentioned":
        value = normalize_shipyard(value)
    else:
        value = normalize_whitespace(value)

    return value or None


def normalize_field_values(field: str, raw_value: Any) -> list[str]:
    if raw_value is None:
        return []

    if isinstance(raw_value, list):
        values = raw_value
    else:
        values = [raw_value]

    normalized: list[str] = []
    seen = set()

    for item in values:
        norm = normalize_facet_value(field, item)
        if not norm:
            continue
        if norm not in seen:
            seen.add(norm)
            normalized.append(norm)

    return normalized


def extract_contribution_from_metadata(metadata: dict[str, Any]) -> dict[str, list[str]]:
    contribution: dict[str, list[str]] = {}

    for field in FACET_FIELDS:
        contribution[field] = normalize_field_values(field, metadata.get(field))

    return contribution


def increment_contribution(
    facets: dict[str, dict[str, int]],
    contribution: dict[str, list[str]],
) -> None:
    for field, values in contribution.items():
        if field not in facets:
            facets[field] = {}
        for value in values:
            facets[field][value] = facets[field].get(value, 0) + 1


def decrement_contribution(
    facets: dict[str, dict[str, int]],
    contribution: dict[str, list[str]],
) -> None:
    for field, values in contribution.items():
        if field not in facets:
            continue
        for value in values:
            if value not in facets[field]:
                continue
            facets[field][value] -= 1
            if facets[field][value] <= 0:
                del facets[field][value]
        if not facets[field]:
            del facets[field]


def remove_deleted_sources(
    root_path: str | Path,
    stored_files: dict[str, dict[str, Any]],
    facets: dict[str, dict[str, int]],
    source_contributions: dict[str, dict[str, list[str]]],
) -> tuple[dict[str, dict[str, Any]], bool]:
    root = Path(root_path)
    changed = False

    existing_source_ids = set()
    for path in root.rglob("*"):
        if path.is_file():
            try:
                source_id = get_source_id(path, root)
                existing_source_ids.add(source_id)
            except Exception:
                pass

    stored_source_ids = set(stored_files.keys())
    deleted_source_ids = stored_source_ids - existing_source_ids

    for source_id in deleted_source_ids:
        old_contribution = source_contributions.get(source_id)
        if old_contribution:
            decrement_contribution(facets, old_contribution)
            source_contributions.pop(source_id, None)
        stored_files.pop(source_id, None)
        changed = True

    return stored_files, changed


def build_source_metadata_from_documents(documents: list[Any]) -> dict[str, dict[str, Any]]:
    """
    Build one metadata record per source_id from the rich metadata that was
    reattached to page-level Documents returned by feed_documents().
    """
    source_meta: dict[str, dict[str, Any]] = {}

    for doc in documents:
        metadata = getattr(doc, "metadata", {}) or {}
        source_id = metadata.get("source_id")
        if not source_id:
            continue

        if source_id not in source_meta:
            source_meta[source_id] = metadata

    return source_meta

def build_source_metadata_from_rich_map(
    rich_metadata_map: dict[tuple[str, int], dict[str, Any]]
) -> dict[str, dict[str, Any]]:
    """
    Collapse page-level rich metadata into one metadata record per source_id.
    The rich metadata was computed at the document level and copied onto pages,
    so taking the first page we see for each source_id is enough.
    """
    source_meta: dict[str, dict[str, Any]] = {}

    for (source_id, _page), metadata in rich_metadata_map.items():
        if source_id not in source_meta:
            source_meta[source_id] = metadata

    return source_meta

def update_metadata_facets_for_files(
    rich_metadata_map: dict[tuple[str, int], dict[str, Any]],
    files_to_upsert: list[str | Path],
    docs_root: str | Path,
    stored_files: dict[str, dict[str, Any]],
    stored_files_path: Path,
    facets_path: Path,
    source_contributions_path: Path,
) -> None:
    facets = load_facets(facets_path)
    source_contributions = load_source_contributions(source_contributions_path)

    # Handle deleted files first.
    stored_files, deleted_any = remove_deleted_sources(
        root_path=docs_root,
        stored_files=stored_files,
        facets=facets,
        source_contributions=source_contributions,
    )

    if not files_to_upsert and not deleted_any:
        print("No facet updates needed.")
        return

    source_metadata = build_source_metadata_from_rich_map(rich_metadata_map)

    for source_id, metadata in source_metadata.items():
        old_contribution = source_contributions.get(source_id)
        if old_contribution:
            decrement_contribution(facets, old_contribution)

        new_contribution = extract_contribution_from_metadata(metadata)
        increment_contribution(facets, new_contribution)
        source_contributions[source_id] = new_contribution

    # Persist all three files so deletes and updates stay in sync.
    _save_json(stored_files_path, stored_files)
    _save_json(facets_path, facets)
    _save_json(source_contributions_path, source_contributions)

    print(f"Saved stored files to: {stored_files_path}")
    print(f"Saved metadata facets to: {facets_path}")
    print(f"Saved source contributions to: {source_contributions_path}")


def initialize_empty_metadata_files(
    facets_path: Path,
    source_contributions_path: Path,
) -> None:
    if not facets_path.exists():
        _save_json(facets_path, {})
    if not source_contributions_path.exists():
        _save_json(source_contributions_path, {})