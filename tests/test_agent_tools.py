"""论文检索 Tool 的纯包装测试。"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

pytest.importorskip("langchain_core", reason="需要安装项目运行依赖")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import agent_tools  # noqa: E402


def create_test_tool(monkeypatch, retrieval_result):
    def fake_retrieve_ranked_evidence(*, query, runtime, reranker, rag_backend):
        assert query
        assert runtime == {"name": "fake_runtime"}
        assert reranker == {"name": "fake_reranker"}
        assert rag_backend == "fake_backend"
        return retrieval_result

    monkeypatch.setattr(
        agent_tools,
        "retrieve_ranked_evidence",
        fake_retrieve_ranked_evidence,
    )
    return agent_tools.create_retrieve_paper_evidence_tool(
        runtime={"name": "fake_runtime"},
        reranker={"name": "fake_reranker"},
        rag_backend="fake_backend",
    )


def test_tool_only_exposes_query(monkeypatch):
    tool = create_test_tool(
        monkeypatch,
        {
            "query": "测试问题",
            "candidate_count": 0,
            "reranked_count": 0,
            "cited_evidence": [],
            "evidence_status": {
                "should_refuse": True,
                "reason": "没有证据",
                "top_similarity": None,
            },
        },
    )
    assert tool.name == "retrieve_paper_evidence"
    assert set(tool.args.keys()) == {"query"}


def test_tool_returns_success_evidence(monkeypatch):
    tool = create_test_tool(
        monkeypatch,
        {
            "query": "论文使用了哪些数据集？",
            "cited_evidence": [
                {
                    "citation_id": "S1",
                    "source_file": "paper.pdf",
                    "page_number": 8,
                    "chunk_id": "12",
                    "similarity": 0.72,
                    "reranker_score": 5.2,
                    "content": "实验使用了 CIFAR-10。",
                }
            ],
            "candidate_count": 20,
            "reranked_count": 20,
            "evidence_status": {
                "should_refuse": False,
                "reason": "证据充分",
                "top_similarity": 0.72,
            },
        },
    )
    result = tool.invoke({"query": "论文使用了哪些数据集？"})
    assert result["status"] == "success"
    assert result["evidence_count"] == 1
    assert "citation_id" not in result["evidence"][0]
    assert result["evidence"][0]["chunk_id"] == 12


def test_tool_rejects_evidence_without_stable_chunk_id(monkeypatch):
    tool = create_test_tool(
        monkeypatch,
        {
            "query": "测试问题",
            "cited_evidence": [
                {
                    "chunk_id": "chunk_12",
                    "content": "无法建立稳定身份的证据。",
                }
            ],
            "candidate_count": 1,
            "reranked_count": 1,
            "evidence_status": {
                "should_refuse": False,
                "reason": "相似度足够",
                "top_similarity": 0.72,
            },
        },
    )
    result = tool.invoke({"query": "测试问题"})
    assert result["status"] == "insufficient_evidence"
    assert result["evidence"] == []


def test_tool_returns_insufficient_evidence(monkeypatch):
    tool = create_test_tool(
        monkeypatch,
        {
            "query": "今天天气怎么样？",
            "cited_evidence": [],
            "candidate_count": 1,
            "reranked_count": 0,
            "evidence_status": {
                "should_refuse": True,
                "reason": "最高相似度低于问题级拒答阈值",
                "top_similarity": 0.21,
            },
        },
    )
    result = tool.invoke({"query": "今天天气怎么样？"})
    assert result["status"] == "insufficient_evidence"
    assert result["evidence"] == []
