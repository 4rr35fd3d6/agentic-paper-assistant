"""论文 LangGraph Agent 的共享状态。"""

from __future__ import annotations

from typing import Any, Literal
from typing_extensions import TypedDict


QueryType = Literal[
    "unknown",
    "fact",
    "method_workflow",
    "comparison",
    "page_read",
    "out_of_scope",
]


class AgentState(TypedDict):
    query: str
    raw_query: str
    resolved_query: str

    current_topic: str | None
    last_query: str | None
    last_answer: str | None

    query_type: QueryType
    target_page: int | None
    rewritten_query: str | None
    sub_queries: list[str]
    comparison_targets: list[str]

    evidence: list[dict[str, Any]]
    evidence_sufficient: bool
    evidence_check_reason: str | None
    seen_chunk_ids: list[int]
    new_evidence_count: int
    tool_call_count: int
    retrieval_round_count: int
    llm_call_count: int
    retry_count: int

    generated_claims: list[dict[str, Any]]
    accepted_claims: list[dict[str, Any]]
    rejected_claims: list[dict[str, Any]]
    claim_validation_reason: str | None
    answer_retry_count: int
    citation_map: dict[int, str]

    answer: str
    citation_valid: bool
    citation_validation_reason: str | None
    refusal_reason: str | None
    execution_trace: list[str]


def create_initial_state(query: str) -> AgentState:
    clean_query = query.replace("\ue000", "").strip()
    if not clean_query:
        raise ValueError("query 不能为空。")

    return {
        "query": clean_query,
        "raw_query": clean_query,
        "resolved_query": clean_query,
        "current_topic": None,
        "last_query": None,
        "last_answer": None,
        "query_type": "unknown",
        "target_page": None,
        "rewritten_query": None,
        "sub_queries": [],
        "comparison_targets": [],
        "evidence": [],
        "evidence_sufficient": False,
        "evidence_check_reason": None,
        "seen_chunk_ids": [],
        "new_evidence_count": 0,
        "tool_call_count": 0,
        "retrieval_round_count": 0,
        "llm_call_count": 0,
        "retry_count": 0,
        "generated_claims": [],
        "accepted_claims": [],
        "rejected_claims": [],
        "claim_validation_reason": None,
        "answer_retry_count": 0,
        "citation_map": {},
        "answer": "",
        "citation_valid": False,
        "citation_validation_reason": None,
        "refusal_reason": None,
        "execution_trace": [],
    }


def create_turn_input(query: str) -> dict[str, str]:
    clean_query = query.replace("\ue000", "").strip()
    if not clean_query:
        raise ValueError("query 不能为空。")
    return {
        "raw_query": clean_query,
        "query": clean_query,
    }
