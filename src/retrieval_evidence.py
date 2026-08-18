"""检索证据的去重、轮次合并与重试保留策略。"""

from __future__ import annotations

from typing import Any

from claim_grounding import get_evidence_text, normalize_chunk_id


def evidence_identity(item: dict[str, Any]) -> tuple[str, Any] | None:
    """返回证据的稳定去重标识。"""
    chunk_id = normalize_chunk_id(item.get("chunk_id"))
    if chunk_id is not None:
        return ("chunk", chunk_id)
    text = get_evidence_text(item)
    return ("text", text) if text else None


def deduplicate_evidence(
    items: list[dict[str, Any]],
    *,
    limit: int = 10,
) -> list[dict[str, Any]]:
    """优先按 chunk_id 去重，缺少编号时按正文去重。"""
    result: list[dict[str, Any]] = []
    seen: set[tuple[str, Any]] = set()

    for item in items:
        identity = evidence_identity(item)
        if identity is None or identity in seen:
            continue
        seen.add(identity)

        normalized = dict(item)
        if identity[0] == "chunk":
            normalized["chunk_id"] = int(identity[1])
        normalized.pop("citation_id", None)
        result.append(normalized)
        if len(result) >= limit:
            break

    return result


def select_balanced_previous_evidence(
    items: list[dict[str, Any]],
    *,
    limit: int,
) -> list[dict[str, Any]]:
    """比较问题重试时，尽量同时保留两侧旧证据。"""
    if limit <= 0:
        return []

    unique_items = deduplicate_evidence(items, limit=max(len(items), 1))
    side_groups: dict[int, list[dict[str, Any]]] = {1: [], 2: []}
    ungrouped: list[dict[str, Any]] = []

    for item in unique_items:
        side = item.get("comparison_side")
        if side in side_groups:
            side_groups[int(side)].append(item)
        else:
            ungrouped.append(item)

    selected: list[dict[str, Any]] = []
    while len(selected) < limit and any(side_groups.values()):
        for side in (1, 2):
            if side_groups[side] and len(selected) < limit:
                selected.append(side_groups[side].pop(0))

    for item in ungrouped:
        if len(selected) >= limit:
            break
        selected.append(item)

    if len(selected) < limit:
        selected_ids = {
            identity
            for identity in (evidence_identity(item) for item in selected)
            if identity is not None
        }
        for item in unique_items:
            identity = evidence_identity(item)
            if identity is None or identity in selected_ids:
                continue
            selected.append(item)
            selected_ids.add(identity)
            if len(selected) >= limit:
                break

    return deduplicate_evidence(selected, limit=limit)


def merge_retry_evidence(
    previous: list[dict[str, Any]],
    new_items: list[dict[str, Any]],
    *,
    limit: int = 10,
    reserved_new: int = 5,
) -> list[dict[str, Any]]:
    """第二轮为新证据预留位置，并保留有代表性的旧证据。"""
    new_kept = deduplicate_evidence(
        new_items,
        limit=min(reserved_new, limit),
    )
    previous_limit = max(0, limit - len(new_kept))
    previous_kept = select_balanced_previous_evidence(
        previous,
        limit=previous_limit,
    )
    return deduplicate_evidence(new_kept + previous_kept, limit=limit)


def count_retained_new_evidence(
    merged: list[dict[str, Any]],
    new_items: list[dict[str, Any]],
) -> int:
    """统计第二轮新增证据中实际进入最终证据集的数量。"""
    new_ids = {
        identity
        for identity in (evidence_identity(item) for item in new_items)
        if identity is not None
    }
    return sum(
        1
        for item in merged
        if evidence_identity(item) in new_ids
    )
