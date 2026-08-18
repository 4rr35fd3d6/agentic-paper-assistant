"""重构后的检索轮次、证据保留和 NLI 截断策略离线测试。"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from retrieval_evidence import merge_retry_evidence  # noqa: E402
from nli_service import LocalNLIModel  # noqa: E402
from agent_nodes import (  # noqa: E402
    resolve_memory_topic,
    resolve_query_pronouns,
)


def test_retry_merge_reserves_new_evidence_and_keeps_both_comparison_sides():
    previous = [
        {
            "chunk_id": index,
            "content": f"旧证据{index}",
            "comparison_side": 1 if index <= 5 else 2,
        }
        for index in range(1, 11)
    ]
    new_items = [
        {"chunk_id": index, "content": f"新证据{index}"}
        for index in range(101, 106)
    ]

    merged = merge_retry_evidence(
        previous,
        new_items,
        limit=10,
        reserved_new=5,
    )

    merged_ids = {item["chunk_id"] for item in merged}
    assert {101, 102, 103, 104, 105}.issubset(merged_ids)
    kept_old = [item for item in merged if item["chunk_id"] <= 10]
    assert len(kept_old) == 5
    assert {item["comparison_side"] for item in kept_old} == {1, 2}


class RecordingTokenizer:
    def __init__(self):
        self.kwargs = None

    def __call__(self, premises, hypotheses, **kwargs):
        self.kwargs = kwargs
        batch = len(premises)
        return {
            "input_ids": torch.ones((batch, 4), dtype=torch.long),
            "attention_mask": torch.ones((batch, 4), dtype=torch.long),
        }


class FakeModel:
    config = SimpleNamespace(id2label={}, label2id={})

    def __call__(self, **kwargs):
        batch = kwargs["input_ids"].shape[0]
        return SimpleNamespace(
            logits=torch.tensor([[5.0, 1.0, 0.0]] * batch)
        )


def test_nli_preserves_claim_and_only_truncates_premise():
    tokenizer = RecordingTokenizer()
    model = LocalNLIModel(device="cpu")
    model._tokenizer = tokenizer
    model._model = FakeModel()
    model._label_to_index = {
        "entailment": 0,
        "neutral": 1,
        "contradiction": 2,
    }

    result = model.predict_many([("很长的证据", "必须完整保留的结论")])

    assert result[0].label == "entailment"
    assert tokenizer.kwargs["truncation"] == "only_first"


def test_pronoun_resolution_does_not_damage_other_words():
    assert resolve_query_pronouns(
        "与其它方法相比如何？",
        "AACFL",
    ) == "与其它方法相比如何？"

    assert resolve_query_pronouns(
        "它们有什么区别？",
        "AACFL",
    ) == "它们有什么区别？"

    assert resolve_query_pronouns(
        "它和FedAvg有什么区别？",
        "AACFL",
    ) == "AACFL和FedAvg有什么区别？"

    assert resolve_query_pronouns(
        "它有什么核心机制？",
        "AACFL",
    ) == "AACFL有什么核心机制？"

    assert resolve_query_pronouns(
        "该方法有什么优势？",
        "AACFL",
    ) == "AACFL有什么优势？"

    assert resolve_query_pronouns(
        "这个算法如何训练？",
        "AACFL",
    ) == "AACFL如何训练？"


def test_memory_topic_updates_only_after_grounded_answer():
    current_topic, detected_topic = (
        resolve_memory_topic(
            resolved_query="AACFL有什么核心机制？",
            previous_topic=None,
            query_type="fact",
            citation_valid=True,
            accepted_claims=[
                {
                    "text": "AACFL是一种自动调整聚类联邦学习框架。"
                }
            ],
        )
    )

    assert current_topic == "AACFL"
    assert detected_topic == "AACFL"

    current_topic, detected_topic = (
        resolve_memory_topic(
            resolved_query="ChatGPT的天气怎么样？",
            previous_topic="AACFL",
            query_type="out_of_scope",
            citation_valid=False,
            accepted_claims=[],
        )
    )

    assert current_topic == "AACFL"
    assert detected_topic is None

    current_topic, detected_topic = (
        resolve_memory_topic(
            resolved_query="FedAvg有什么特点？",
            previous_topic="AACFL",
            query_type="fact",
            citation_valid=False,
            accepted_claims=[],
        )
    )

    assert current_topic == "AACFL"
    assert detected_topic is None

    current_topic, detected_topic = (
        resolve_memory_topic(
            resolved_query="FedAvg有什么特点？",
            previous_topic="AACFL",
            query_type="fact",
            citation_valid=True,
            accepted_claims=[],
        )
    )

    assert current_topic == "AACFL"
    assert detected_topic is None

