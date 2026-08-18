"""Agentic RAG 的论文检索工具包装。"""

from __future__ import annotations

from typing import Any

from langchain_core.tools import tool

from claim_grounding import normalize_chunk_id
from rag_service import retrieve_ranked_evidence


def create_retrieve_paper_evidence_tool(
    *,
    runtime: dict[str, Any],
    reranker: dict[str, Any],
    rag_backend: Any,
):
    """创建只暴露 query 参数的论文证据检索工具。"""

    @tool
    def retrieve_paper_evidence(query: str) -> dict[str, Any]:
        """从当前论文中检索与问题直接相关的证据。"""
        retrieval_data = retrieve_ranked_evidence(
            query=query,
            runtime=runtime,
            reranker=reranker,
            rag_backend=rag_backend,
        )
        evidence_status = retrieval_data["evidence_status"]

        if evidence_status["should_refuse"]:
            return {
                "status": "insufficient_evidence",
                "query": retrieval_data["query"],
                "reason": evidence_status["reason"],
                "top_similarity": evidence_status["top_similarity"],
                "evidence": [],
            }

        evidence_items: list[dict[str, Any]] = []
        for result in retrieval_data["cited_evidence"]:
            chunk_id = normalize_chunk_id(result.get("chunk_id"))
            content = str(result.get("content") or "").strip()
            if chunk_id is None or not content:
                continue

            evidence_items.append(
                {
                    # Agent 内部只使用稳定 chunk_id；S 编号由 Python 最后生成。
                    "source_file": result.get("source_file"),
                    "page_number": result.get("page_number"),
                    "chunk_id": chunk_id,
                    "similarity": result.get("similarity"),
                    "reranker_score": result.get("reranker_score"),
                    "content": content,
                }
            )

        if not evidence_items:
            return {
                "status": "insufficient_evidence",
                "query": retrieval_data["query"],
                "reason": "检索结果缺少可验证的 Chunk ID 或正文。",
                "top_similarity": evidence_status["top_similarity"],
                "evidence": [],
            }

        return {
            "status": "success",
            "query": retrieval_data["query"],
            "top_similarity": evidence_status["top_similarity"],
            "candidate_count": retrieval_data["candidate_count"],
            "reranked_count": retrieval_data["reranked_count"],
            "evidence_count": len(evidence_items),
            "evidence": evidence_items,
        }

    return retrieve_paper_evidence
