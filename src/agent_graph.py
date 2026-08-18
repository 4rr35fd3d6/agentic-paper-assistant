"""论文 LangGraph Agent 的工作流组装。"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from agent_nodes import create_agent_nodes
from agent_routes import (
    route_after_claim_validation,
    route_after_classify,
    route_after_decompose,
    route_after_evidence_check,
)
from agent_state import AgentState


def create_agent_graph(
    runtime,
    reranker,
    rag_backend,
    *,
    nli_model=None,
    checkpointer=None,
):
    """使用当前论文运行环境创建正式 LangGraph。"""
    nodes = create_agent_nodes(
        runtime=runtime,
        reranker=reranker,
        rag_backend=rag_backend,
        nli_model=nli_model,
    )

    builder = StateGraph(AgentState)

    for node_name in (
        "prepare_turn",
        "classify_query",
        "read_page",
        "decompose_query",
        "multi_retrieve",
        "retrieve",
        "evidence_check",
        "rewrite_query",
        "generate_claims",
        "validate_claims",
        "prepare_answer_retry",
        "render_answer",
        "refuse",
        "save_turn_memory",
    ):
        builder.add_node(node_name, nodes[node_name])

    builder.add_edge(START, "prepare_turn")
    builder.add_edge("prepare_turn", "classify_query")
    builder.add_conditional_edges("classify_query", route_after_classify)
    builder.add_conditional_edges("decompose_query", route_after_decompose)

    builder.add_edge("read_page", "evidence_check")
    builder.add_edge("multi_retrieve", "evidence_check")
    builder.add_edge("retrieve", "evidence_check")
    builder.add_conditional_edges("evidence_check", route_after_evidence_check)

    builder.add_edge("rewrite_query", "retrieve")
    builder.add_edge("generate_claims", "validate_claims")
    builder.add_conditional_edges(
        "validate_claims",
        route_after_claim_validation,
    )
    builder.add_edge("prepare_answer_retry", "generate_claims")
    builder.add_edge("render_answer", "save_turn_memory")
    builder.add_edge("refuse", "save_turn_memory")
    builder.add_edge("save_turn_memory", END)

    return builder.compile(checkpointer=checkpointer)
