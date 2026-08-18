"""LangGraph 条件路由。"""

from __future__ import annotations

from typing import Literal
from agent_state import AgentState


MAX_QUERY_REWRITES = 1
MAX_RETRIEVAL_ROUNDS = 2
MAX_ANSWER_RETRIES = 1


def route_after_classify(
    state: AgentState,
) -> Literal["retrieve", "read_page", "decompose_query", "refuse"]:
    query_type = state.get("query_type", "unknown")
    if query_type == "out_of_scope":
        return "refuse"
    if query_type == "page_read":
        return "read_page"
    if query_type == "comparison":
        return "decompose_query"
    return "retrieve"


def route_after_decompose(
    state: AgentState,
) -> Literal["multi_retrieve", "refuse"]:
    return "multi_retrieve" if len(state.get("sub_queries", [])) == 2 else "refuse"


def route_after_evidence_check(
    state: AgentState,
) -> Literal["generate_claims", "rewrite_query", "refuse"]:
    if state.get("evidence_sufficient", False):
        return "generate_claims"

    if state.get("query_type") == "page_read":
        return "refuse"

    if (
        state.get("retry_count", 0) < MAX_QUERY_REWRITES
        and state.get("retrieval_round_count", 0) < MAX_RETRIEVAL_ROUNDS
    ):
        return "rewrite_query"
    return "refuse"


def route_after_claim_validation(
    state: AgentState,
) -> Literal["render_answer", "prepare_answer_retry", "refuse"]:
    if state.get("accepted_claims"):
        return "render_answer"
    if state.get("answer_retry_count", 0) < MAX_ANSWER_RETRIES:
        return "prepare_answer_retry"
    return "refuse"
