import sys
from pathlib import Path

import pytest

pytest.importorskip("openai", reason="需要安装项目运行依赖")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PRACTICE_DIR = PROJECT_ROOT / "practice"

if str(PRACTICE_DIR) not in sys.path:
    sys.path.insert(
        0,
        str(PRACTICE_DIR)
    )


from day22_reranker_rag import (
    evaluate_evidence_sufficiency,
    validate_answer_citations,
)


def test_empty_results_should_refuse():
    result = evaluate_evidence_sufficiency(
        results=[],
        refusal_threshold=0.36
    )

    assert result["should_refuse"] is True
    assert result["top_similarity"] is None


def test_similarity_below_threshold_should_refuse():
    result = evaluate_evidence_sufficiency(
        results=[
            {
                "similarity": 0.35
            }
        ],
        refusal_threshold=0.36
    )

    assert result["should_refuse"] is True
    assert result["top_similarity"] == pytest.approx(
        0.35
    )


def test_similarity_reaching_threshold_should_answer():
    result = evaluate_evidence_sufficiency(
        results=[
            {
                "similarity": 0.31
            },
            {
                "similarity": 0.42
            }
        ],
        refusal_threshold=0.36
    )

    assert result["should_refuse"] is False
    assert result["top_similarity"] == pytest.approx(
        0.42
    )


def test_valid_citation_should_pass():
    result = validate_answer_citations(
        answer=(
            "论文采用了基于语义表示的"
            "检索方法 [S1]。"
        ),
        evidence=[
            {
                "citation_id": "S1"
            },
            {
                "citation_id": "S2"
            }
        ]
    )

    assert result["citation_valid"] is True
    assert result["has_citations"] is True
    assert result["invalid_citation_ids"] == []
    assert result["used_citation_ids"] == [
        "S1"
    ]


def test_unknown_citation_should_fail():
    result = validate_answer_citations(
        answer=(
            "论文使用了某种方法 [S3]。"
        ),
        evidence=[
            {
                "citation_id": "S1"
            },
            {
                "citation_id": "S2"
            }
        ]
    )

    assert result["citation_valid"] is False
    assert result["has_citations"] is True
    assert result["invalid_citation_ids"] == [
        "S3"
    ]


def test_answer_without_citation_should_fail():
    result = validate_answer_citations(
        answer="论文使用了某种检索方法。",
        evidence=[
            {
                "citation_id": "S1"
            }
        ]
    )

    assert result["citation_valid"] is False
    assert result["has_citations"] is False
    assert result["invalid_citation_ids"] == []
