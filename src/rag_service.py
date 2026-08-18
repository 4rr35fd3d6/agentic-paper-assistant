"""
经典 RAG 和 Agentic RAG 共用的论文检索服务。

本模块只负责：
Query → FAISS → Reranker → 证据筛选 → 拒答判断

本模块不调用大语言模型，也不负责生成最终答案。
"""

from __future__ import annotations

from typing import Any


def retrieve_ranked_evidence(
    *,
    query: str,
    runtime: dict[str, Any],
    reranker: dict[str, Any],
    rag_backend: Any,
) -> dict[str, Any]:
    """
    执行论文证据检索。

    参数：
        query：
            用户问题。

        runtime：
            当前论文知识库运行对象，包含：
            chunks、embedding_model、index 等。

        reranker：
            已加载的 BGE Reranker 资源。

        rag_backend：
            原项目中的 day22_reranker_rag 模块。

    返回：
        候选证据、重排结果、最终证据和拒答状态。
    """
    cleaned_query = query.strip()

    if not cleaned_query:
        raise ValueError("检索问题不能为空。")

    # 第一步：使用 Embedding 和 FAISS 召回 Top-20 候选。
    candidate_results = rag_backend.retrieve_top_chunks(
        query=cleaned_query,
        chunks=runtime["chunks"],
        model=runtime["embedding_model"],
        index=runtime["index"],
        top_k=rag_backend.RETRIEVAL_CANDIDATE_K,
        min_similarity=rag_backend.CHUNK_MIN_SIMILARITY,
    )

    # 第二步：使用 Cross-Encoder 联合读取 Query 和 Chunk，
    # 对候选结果重新排序。
    reranked_results = rag_backend.rerank_candidates(
        query=cleaned_query,
        candidates=candidate_results,
        tokenizer=reranker["tokenizer"],
        model=reranker["model"],
        device=reranker["device"],
    )

    # 第三步：去除参考文献、目录、重复文本等噪声，
    # 并限制最终证据数量。
    raw_selected_results = (
        rag_backend.select_reranked_evidence(
            reranked_results
        )
    )

    # 使用用户上传时的原始论文名称，
    # 不把内部哈希文件名暴露给回答和页面。
    selected_results = []

    for result in raw_selected_results:
        normalized_result = dict(result)

        normalized_result["source_file"] = runtime[
            "original_filename"
        ]

        selected_results.append(normalized_result)

    # 拒答判断仍然基于原始 FAISS 候选相似度，
    # 保持与经典 RAG 当前逻辑一致。
    evidence_status = (
        rag_backend.evaluate_evidence_sufficiency(
            results=candidate_results,
            refusal_threshold=(
                rag_backend.QUERY_REFUSAL_THRESHOLD
            ),
        )
    )

    if evidence_status["should_refuse"]:
        cited_evidence = []
    else:
        cited_evidence = (
            rag_backend.prepare_cited_evidence(
                selected_results[
                    :rag_backend.FINAL_EVIDENCE_COUNT
                ]
            )
        )

    return {
        "query": cleaned_query,
        "candidate_results": candidate_results,
        "reranked_results": reranked_results,
        "selected_results": selected_results,
        "cited_evidence": cited_evidence,
        "evidence_status": evidence_status,
        "candidate_count": len(candidate_results),
        "reranked_count": len(reranked_results),
        "selected_count": len(selected_results),
    }